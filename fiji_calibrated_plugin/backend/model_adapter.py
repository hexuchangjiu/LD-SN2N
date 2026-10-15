"""Standalone, fixed-chart inference for the hash-frozen SN2N AUnet only.

No training code, old project import, automatic contrast fitting, tile blending,
or output clipping is used. Trust enables legacy pickle deserialization, which
is not a security sandbox and must only be used for a user-controlled file.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import pickle
import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .vendor import aunet


SOURCE_ORIGIN = (
    "E31/20260910/source_snapshot/SN2N_main/SelfN2N/AUnet.py"
)
SOURCE_SHA256 = "23846ae4f5c358d43999a2b4837cd87949a1fb8d2b3e30b680dea9d380d8f330"
INFERENCE_ORIGIN = "experiments/e64_biosr_extended/inference_qa.py:full_infer"
INFERENCE_SOURCE_SHA256 = "7120a395324624f821fe20c8bc326b9717d19d43d4c8e3350bda07ed93368a9c"
HALO = 32
_LEGACY_MODULES = {
    "AUnet", "SelfN2N.AUnet", "SN2N_main.SelfN2N.AUnet",
    "SN2N.SelfN2N.AUnet", aunet.__name__,
}
_LEGACY_CLASSES = {name: getattr(aunet, name) for name in
                   ("AUnet", "DoubleConv", "Down", "Up", "OutConv")}


class CheckpointTrustError(ValueError):
    """A checkpoint cannot be safely loaded without explicit user trust."""


class MissingChartError(ValueError):
    """No checkpoint chart or caller-supplied fixed numeric chart exists."""


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode() + str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _check_vendor():
    path = Path(aunet.__file__).resolve()
    # Exact E31 snapshot; tolerate only transport line endings / trailing blanks.
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip() + "\n"
    if hashlib.sha256(normalized.encode("utf-8")).hexdigest() != SOURCE_SHA256:
        raise ValueError("Vendored AUnet source changed; requalification is required")
    return _file_sha256(path)


class _MappedLegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in _LEGACY_MODULES and name in _LEGACY_CLASSES:
            return _LEGACY_CLASSES[name]
        # Needed by historical state containers (including NumPy RNG state).
        # This is a compatibility restriction, not a secure pickle sandbox.
        if module in {"builtins", "__builtin__", "collections", "_codecs", "numpy"} or \
                module == "torch" or module.startswith(("torch.", "numpy.")):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Unsupported legacy source class: {module}.{name}")


_LEGACY_PICKLE = SimpleNamespace(
    __name__="fiji_aunet_legacy_pickle", Unpickler=_MappedLegacyUnpickler,
    load=pickle.load, loads=pickle.loads, dump=pickle.dump, dumps=pickle.dumps,
)


def _load_payload(path, trust):
    version = re.match(r'^(\d+)\.(\d+)', torch.__version__)
    if not trust and (version is None or tuple(map(int,version.groups())) < (2,6)):
        raise CheckpointTrustError('Weights-only mode requires PyTorch >=2.6. Upgrade PyTorch; do not enable legacy trust for an unknown file.')
    try:
        return torch.load(path, map_location="cpu", weights_only=True), "weights_only"
    except (pickle.UnpicklingError, RuntimeError) as error:
        if not trust:
            raise CheckpointTrustError(
                "Checkpoint was rejected by weights_only=True. Legacy checkpoints "
                "require trust_checkpoint=True only for a file you control."
            ) from error
    return torch.load(path, map_location="cpu", weights_only=False,
                      pickle_module=_LEGACY_PICKLE), "trusted_legacy_pickle"


def _module_signature(model):
    return [(name, type(module), module.extra_repr())
            for name, module in model.named_modules()]


def _extract_state(payload, canonical):
    metadata = payload if isinstance(payload, Mapping) else {}
    candidate = payload
    if isinstance(payload, Mapping):
        present = [key for key in ("model_state_dict", "state_dict", "model") if key in payload]
        if len(present) > 1:
            raise ValueError("Ambiguous checkpoint: multiple model representations")
        if present:
            candidate = payload[present[0]]
    if isinstance(candidate, torch.nn.Module):
        if type(candidate) is not aunet.AUnet or \
                (candidate.n_channels, candidate.n_classes, candidate.bilinear) != (1, 1, True) or \
                _module_signature(candidate) != _module_signature(canonical):
            raise ValueError("Only the frozen AUnet(1,1,True), base64 architecture is supported")
        for module in candidate.modules():
            if any(getattr(module, name, {}) for name in (
                    "_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
                    "_state_dict_hooks", "_state_dict_pre_hooks")):
                raise ValueError("Legacy module hooks are not supported")
        candidate = candidate.state_dict()
    if not isinstance(candidate, Mapping) or not candidate:
        raise ValueError("Checkpoint must contain an AUnet state_dict")
    expected = canonical.state_dict()
    if set(candidate) != set(expected):
        raise ValueError("Checkpoint tensor keys do not match the frozen AUnet architecture")
    for name, expected_tensor in expected.items():
        value = candidate[name]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or \
                value.shape != expected_tensor.shape or value.dtype != expected_tensor.dtype:
            raise ValueError(f"Unsupported AUnet tensor shape/dtype: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        if name.endswith("running_var") and bool((value < 0).any()):
            raise ValueError(f"Negative BatchNorm variance: {name}")
    return candidate, metadata


def _validated_chart(value):
    if not isinstance(value, Mapping) or "offset" not in value or "scale" not in value:
        raise ValueError("A fixed chart requires explicit offset and scale")
    offset, scale = float(value["offset"]), float(value["scale"])
    if not math.isfinite(offset) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Chart offset must be finite and scale finite and positive")
    return {"offset": offset, "scale": scale}


def _resolve_chart(metadata, requested):
    declared = metadata.get("chart")
    if declared is not None:
        stored = _validated_chart(declared)
        if requested is not None and _validated_chart(requested) != stored:
            raise ValueError("Manual chart conflicts with the checkpoint-declared chart")
        known_training_chart = metadata.get("schema") in {
            "e63-inference-only-v1", "e64-inference-only-v1",
        }
        return stored, "checkpoint", known_training_chart
    if requested is None:
        raise MissingChartError(
            "Checkpoint has no numerical training chart; supply chart={offset, scale}. "
            "No chart is inferred from the query image."
        )
    return _validated_chart(requested), "manual_unqualified", False


def _resolve_device(requested):
    name = str(requested)
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are auto, cpu, and cuda[:index]")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was explicitly requested but is unavailable; use device='auto' or 'cpu'")
    return result


class AUnetPredictor:
    """Frozen native-unit predictor. Each call uses exactly one full-field pass."""

    def __init__(self, model, chart, manifest):
        self._model = model.eval().requires_grad_(False)
        self._chart = dict(chart)
        self._manifest = manifest
        self._initial_state_sha256 = _state_sha256(model)
        self._initial_module_signature = _module_signature(model)
        self._manifest["model_state_sha256"] = self._initial_state_sha256
        self._manifest["state_tensor_count"] = len(model.state_dict())

    @property
    def manifest(self):
        """JSON-serializable provenance; callers cannot mutate predictor state."""
        return copy.deepcopy(self._manifest)

    def assert_unchanged(self):
        if any(module.training for module in self._model.modules()):
            raise AssertionError("Predictor must remain in frozen eval mode")
        if any(parameter.requires_grad for parameter in self._model.parameters()):
            raise AssertionError("Predictor gradients were enabled")
        if _state_sha256(self._model) != self._initial_state_sha256:
            raise AssertionError("Model parameters or BatchNorm buffers changed")
        if _module_signature(self._model) != self._initial_module_signature:
            raise AssertionError("Model layer configuration changed")
        if _file_sha256(self._manifest["checkpoint_path"]) != self._manifest["checkpoint_sha256"]:
            raise AssertionError("Checkpoint file changed")
        return True

    def predict(self, native2d):
        array = np.asarray(native2d)
        if array.ndim != 2 or min(array.shape) < 2 or array.dtype.kind not in "uif":
            raise ValueError("Expected a real numeric 2D image with both dimensions >= 2")
        native = np.asarray(array, dtype=np.float32)
        if not np.isfinite(native).all():
            raise ValueError("Input must be finite in FP32 coordinates")
        # This chart is fixed when loading, never fitted again on a loop state.
        image = ((native - self._chart["offset"]) / self._chart["scale"]).astype(np.float32)
        if not np.isfinite(image).all():
            raise ValueError("The fixed chart produces nonfinite FP32 model coordinates")
        if any(module.training for module in self._model.modules()):
            raise AssertionError("Predictor must remain in frozen eval mode")
        h, w = image.shape
        hh = 16 * math.ceil((h + 2 * HALO) / 16)
        ww = 16 * math.ceil((w + 2 * HALO) / 16)
        padded = np.pad(image, ((HALO, hh-h-HALO), (HALO, ww-w-HALO)), mode="reflect")
        device = next(self._model.parameters()).device
        # Explicitly disable ambient autocast; no AMP/TF32/tiled alternative.
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            tensor = torch.from_numpy(padded.copy())[None, None].to(device)
            result = self._model(tensor)[0, 0, HALO:HALO+h, HALO:HALO+w].detach().cpu().numpy().copy()
        restored = (self._chart["offset"] + self._chart["scale"] * result).astype(np.float32)
        if restored.shape != native.shape or not np.isfinite(restored).all():
            raise FloatingPointError("Invalid full-field prediction")
        return restored


def load_predictor(checkpoint_path, device="auto", trust_checkpoint=False, chart=None):
    """Load only the exact frozen AUnet state layout, never infer an architecture.

    Safe state_dict containers are loaded with weights_only=True. Historical
    modules/NumPy-bearing containers require explicit trust; old AUnet class
    names are mapped to the bundled source, without importing the old project.
    Missing checkpoint chart -> explicit manual chart -> unqualified manifest.
    E54 state checkpoints therefore also require a manual numeric chart.
    """
    vendor_file_sha = _check_vendor()
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    before = _file_sha256(path)
    payload, load_mode = _load_payload(path, trust_checkpoint)
    # Random initialization is overwritten; do not consume the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        model = aunet.AUnet(1, 1, True)
    state, metadata = _extract_state(payload, model)
    fixed_chart, chart_source, qualified = _resolve_chart(metadata, chart)
    model.load_state_dict(state, strict=True)
    target = _resolve_device(device)
    # Match the FP32 deterministic policy used by E64's trainer/inference.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model.to(device=target, dtype=torch.float32)
    if _file_sha256(path) != before:
        raise RuntimeError("Checkpoint changed while loading")
    manifest = {
        "schema": "fiji-calibrated-aunet-adapter-v1",
        "architecture": "AUnet(1,1,True); base64; native sigmoid",
        "checkpoint_path": str(path), "checkpoint_sha256": before,
        "checkpoint_schema": metadata.get("schema"), "load_mode": load_mode,
        "trust_checkpoint": bool(trust_checkpoint),
        "chart": fixed_chart, "model_chart": fixed_chart, "chart_source": chart_source,
        "model_chart_qualified": qualified,
        "chart_qualification_scope": "Declared E63/E64 training coordinates only; not camera/noise or efficacy qualification",
        "physical_noise_qualified": False, "per_call_normalization": False,
        "output_clipping": False, "device_requested": str(device), "device": str(target),
        "inference": "native_fullframe_frozen_bn", "precision": "FP32", "halo": HALO,
        "deterministic_algorithms": True, "allow_tf32": False,
        "padding": "numpy reflect; round padded shape upward to multiple of 16",
        "vendor_origin": SOURCE_ORIGIN, "vendor_original_sha256": SOURCE_SHA256,
        "vendor_file_sha256": vendor_file_sha,
        "inference_origin": INFERENCE_ORIGIN,
        "inference_origin_sha256": INFERENCE_SOURCE_SHA256,
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
    }
    return AUnetPredictor(model, fixed_chart, manifest)
