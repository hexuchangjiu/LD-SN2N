# Verification

Version 0.5.0 was checked on Windows on 15 October 2026 with Python 3.11.13, PyTorch 2.6.0 CPU and Fiji's JDK 21.0.7. The JAR targets Java 8. These are software checks, not new denoising experiments.

| Check | Result |
|---|---|
| Python tests | 99 passed |
| Java smoke suites | 4 passed: frontend, legacy path compatibility, single-JAR runtime and menu discovery |
| Real Python runtime checks | 32 checks passed, including interpreter fallback, revalidation and timeout |
| Independent builds | Identical JAR bytes across two builds |
| Embedded runtime | All 12 files matched the embedded hash manifest |
| Single-JAR installation | Only the JAR was present; no adjacent backend or configuration was supplied |
| Environment discovery | A Python environment missing dependencies was skipped; a valid environment was selected and remembered |
| Explicit Python override | An invalid selection failed instead of silently switching interpreters |
| Cache recovery | Altered backend code was retained in quarantine and replaced from verified embedded resources |
| Extraction safety | Traversal, duplicate paths, wrong hashes, oversize entries, wrong versions and a real Windows junction were rejected |
| Preflight interruption | A real sleeping process was stopped on timeout and on cancellation |
| Full image pipeline | Java export, automatic runtime selection, calibration, Loop and all output artifacts completed |
| Numerical source | The seven backend modules were byte-identical to the prior tested release |

The isolated image test used 150 frames of 64×80 float32 synthetic data and a deliberately untrained constant-output checkpoint. All source pixels were preserved, an active ROI did not crop the image, and the selected frame index was retained. D1 and every denoised trajectory plane had the expected value 125; the structure check failed and the formal result remained `DEFER`. This tests the pipeline and negative control, not restoration quality.

The independent test runner initially misread Windows console output as UTF-8. Its first run was retained as failed; correcting the runner's decoding allowed the repeated test to finish. No production change was made to obtain that pass.

## Artifact identity

```text
LoopSN2N-0.5.0.jar
Bytes: 99673
SHA256: 684ba9e5c226a20f5c56c143bfdda9b7d9e484340e26d35015b41cbd52357214
```

The JAR includes project source and notices, not Python or PyTorch. A machine-readable summary is provided in [verification_summary.json](verification_summary.json). The previous release's reports and original archives remain under `releases/0.4.0/`.

## Repeating the checks

From `loop-Drift`:

```console
python scripts/verify_release.py --output build/verification-new
python scripts/build_fiji.py --fiji "/path/to/Fiji" --work-dir build/fiji-new --output "../plugin/LoopSN2N-0.5.0.jar"
```

The builder preserves logs and runs the Java smoke suites against the staged JAR. Use a new output directory for each verification run.

## Limits

Interactive GUI operation, macOS/Linux installations and current-release CUDA inference have not been verified. Windows symbolic-link creation was unavailable under the test account; a real directory junction rejection was tested separately. Concurrent extraction was tested, but hostile concurrent cache modification was not.

Dependency discovery has time limits and can be cancelled. Assessment cancellation remains cooperative: there is no guaranteed termination time for a stalled native or GPU operation. The tests do not establish biological efficacy, feedback-state noise validity or independent deployment-gate qualification. Licensing remains unresolved as described in [LICENSE_STATUS.md](../LICENSE_STATUS.md).
