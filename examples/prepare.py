"""Prepare LoopSN2N's Fiji examples from this checkout and the upstream BioSR data."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen
import zipfile


HERE = Path(__file__).resolve().parent
CHUNK = 1024 * 1024
GIB = 1024**3
MT = ("data/BioSR_MT.mrc", 4537096,
      "0aaf8d8ae16a8e46eabdd86dd01e182a4053aae5e56fcb46781eea54bf564b14",
      "https://ndownloader.figshare.com/files/25714514", 1633356198,
      "df983755a1b45d512b000475e3072fc7", "Microtubules/Cell_002/RawSIMData_level_05.mrc")
OTF = ("runtime/raw/recon_assets/TIRF_OTF2D_488.mrc", 3080,
       "e496e544605e9b4a702af028b1cdd946be4ee24e6ba62e845475fda06a8a363a",
       "https://ndownloader.figshare.com/files/43867986", 19792,
       "5136d6396458de71bcfac1b1129c6d0b",
       "Supplementary Files for BioSR/OTF_File/TIRF_OTF2D_488.mrc")
FAIR = ("runtime/raw/recon_assets/fairSIM_plugin.jar", 545730,
        "ce221779b88cda8b071d8a95908f21d4970b30980f107ca923df2f8d2249a956",
        "https://github.com/fairSIM/fairSIM/releases/download/v1.4.1/fairSIM_plugin.jar")


def safe(root, rel):
    p = PurePosixPath(rel)
    if (not p.parts or p.is_absolute() or "\\" in rel or ".." in p.parts or
            PureWindowsPath(rel).drive or any(":" in i for i in p.parts)):
        raise ValueError(f"Invalid relative resource path: {rel}")
    path = root.joinpath(*p.parts)
    for x in (root, *(root.joinpath(*p.parts[:n]) for n in range(1, len(p.parts) + 1))):
        if x.is_symlink() or (x.exists() and getattr(x.lstat(), "st_file_attributes", 0)
                              & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)):
            raise ValueError(f"Refusing linked destination or source: {x}")
    return path


def check(path, size, digest, alg="sha256"):
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"Incorrect size or missing file: {path}")
    h = hashlib.new(alg)
    with path.open("rb") as r:
        for b in iter(lambda: r.read(CHUNK), b""):
            h.update(b)
    if h.hexdigest() != digest:
        raise ValueError(f"{alg} mismatch: {path}")


def copy_stream(stream, path, size):
    total = 0
    with path.open("xb") as w:
        for b in iter(lambda: stream.read(CHUNK), b""):
            total += len(b)
            if total > size:
                raise ValueError(f"Input exceeds expected {size} bytes")
            w.write(b)
    if total != size:
        raise ValueError(f"Expected {size} bytes, received {total}")


def retrieve(url, path, size):
    print(f"Downloading {url} ({size / CHUNK:.1f} MiB)", flush=True)
    request = Request(url, headers={"User-Agent": "LoopSN2N-examples/1", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=90) as r:
        if not r.geturl().startswith("https://"):
            raise ValueError("Download redirected away from HTTPS")
        copy_stream(r, path, size)


def acq(spec, direct, archive, stage, dest):
    rel, size, digest, url, *packed = spec
    out = safe(stage, rel)
    out.parent.mkdir(parents=True, exist_ok=True)
    if direct is not None:
        check(direct, size, digest)
    if archive is not None:
        check(archive, packed[0], packed[1], "md5")
    existing = safe(dest, rel)
    if existing.exists():
        check(existing, size, digest)
        return
    if direct is not None:
        with direct.open("rb") as f:
            copy_stream(f, out, size)
    elif not packed:
        retrieve(url, out, size)
    else:
        packed_size, md5, member = packed
        if archive is None:
            archive = stage / (out.name + ".zip")
            retrieve(url, archive, packed_size)
            check(archive, packed_size, md5, "md5")
        with zipfile.ZipFile(archive) as z:
            hit = [m for m in z.infolist() if m.filename == member]
            if len(hit) != 1 or hit[0].file_size != size:
                raise ValueError(f"Incorrect archive member: {member}")
            with z.open(hit[0]) as f:
                copy_stream(f, out, size)
    check(out, size, digest)


def room(path, needed):
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < needed:
        raise ValueError(f"Need {needed / GIB:.1f} GiB free on {path.anchor}")


def publish(staged, target, size, digest):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        check(target, size, digest)
        return
    try:
        # O_EXCL via a hard link: another installer cannot overwrite an asset.
        os.link(staged, target)
    except FileExistsError:
        check(target, size, digest)
    except OSError:
        # The destination can reside on another drive or a FAT filesystem.
        with staged.open("rb") as f:
            copy_stream(f, target, size)
        check(target, size, digest)


def env(dest, provided):
    if provided is not None:
        python = provided.resolve(strict=True)
    else:
        if sys.version_info[:2] not in ((3, 10), (3, 11)):
            raise ValueError("Use CPython 3.10 or 3.11 to create the pinned environment.")
        venv = safe(dest, "runtime/venv")
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        pending = venv / ".preparing-loopsn2n"
        if not venv.exists():
            venv.mkdir(parents=True)
            pending.touch(exist_ok=False)
        if pending.exists():
            # Only repair environments this command started; never change a user's
            # existing environment following an interrupted wheel download.
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
            pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:"]
            wheel_source = ([] if sys.platform == "darwin" else
                            ["--index-url", "https://download.pytorch.org/whl/cpu"])
            subprocess.run(pip + wheel_source + ["torch==2.6.0"], check=True)
            subprocess.run(pip + ["numpy==1.25.2", "scipy==1.11.1", "tifffile==2023.7.10"], check=True)
            pending.unlink()
    subprocess.run([str(python), "-B", "-c",
                    "import numpy,scipy,torch,tifffile; "
                    "print('Python environment:',numpy.__version__,scipy.__version__,torch.__version__,tifffile.__version__)"],
                   check=True)
    return python


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--example", choices=("both", "simulation", "biosr"), default="both")
    placement = p.add_mutually_exclusive_group()
    placement.add_argument("--fiji", type=Path, help="Fiji installation root (also installs the JAR)")
    placement.add_argument("--dest", type=Path, help="Example directory (default: this checkout)")
    runtime = p.add_mutually_exclusive_group()
    runtime.add_argument("--python", type=Path, help="Use this existing scientific Python without modifying it")
    runtime.add_argument("--files-only", action="store_true", help="Prepare and verify files without installing Python")
    raw = p.add_mutually_exclusive_group()
    raw.add_argument("--mt-zip", type=Path, help="Local copy of the official Microtubules ZIP")
    raw.add_argument("--raw-mrc", type=Path, help="Already extracted Cell_002 L05 raw MRC")
    otf = p.add_mutually_exclusive_group()
    otf.add_argument("--otf-zip", type=Path, help="Local copy of the official supplementary ZIP")
    otf.add_argument("--otf-mrc", type=Path, help="Already extracted 488 nm OTF MRC")
    p.add_argument("--fairsim-jar", type=Path, help="Local copy of the upstream fairSIM v1.4.1 JAR")
    a = p.parse_args()
    jar = None
    if a.fiji:
        fiji = a.fiji.resolve(strict=True)
        if not (fiji / "plugins").is_dir() or not (fiji / "jars").is_dir():
            raise ValueError("--fiji expects the Fiji application directory containing plugins/ and jars/")
        dest = safe(fiji, "plugins/LoopSN2N/examples")
        jar = safe(fiji, "plugins/LoopSN2N_Fiji.jar")
    else:
        dest = (a.dest or HERE).absolute()
    if dest == Path(dest.anchor) or dest.is_symlink() or dest.resolve() != dest:
        raise ValueError(f"Refusing root, linked or non-canonical output directory: {dest}")
    biosr = a.example != "simulation"
    if not biosr and any((a.mt_zip, a.raw_mrc, a.otf_zip, a.otf_mrc, a.fairsim_jar)):
        raise ValueError("BioSR inputs were specified for the simulation example")
    records = json.loads((HERE / "assets.json").read_text(encoding="utf-8"))
    files = [(rel, n, sha) for rel, group, n, sha in records
             if group == "shared" or a.example == "both" or group == a.example]
    upstream = {MT[0], OTF[0], FAIR[0]}
    files = [(rel, n, sha) for rel, n, sha in files if rel not in upstream]
    # Released examples include the original BioSR planes and OTF. Check those bytes
    # against the source identities; a partial checkout may obtain them from Figshare.
    if biosr:
        for spec, direct_arg, archive in ((MT, "raw_mrc", a.mt_zip), (OTF, "otf_mrc", a.otf_zip)):
            source = safe(HERE, spec[0])
            if getattr(a, direct_arg) is None and archive is None and source.exists():
                setattr(a, direct_arg, source)
    for rel, n, sha in files:
        if rel == "../plugin/LoopSN2N_Fiji.jar":
            continue
        check(safe(HERE, rel), n, sha)
        target = safe(dest, rel)
        if target.exists():
            check(target, n, sha)
    missing_mt = biosr and not a.mt_zip and not a.raw_mrc and not safe(dest, MT[0]).exists()
    budget = (MT[4] if missing_mt else 0) + sum(n for _, n, _ in files) + (0 if a.files_only or a.python else 3 * GIB)
    room(dest, budget + 128 * CHUNK)
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=dest) as tmp:
        stage = Path(tmp)
        if biosr:
            acq(MT, a.raw_mrc, a.mt_zip, stage, dest)
            acq(OTF, a.otf_mrc, a.otf_zip, stage, dest)
            acq(FAIR, a.fairsim_jar, None, stage, dest)
        for rel, n, sha in files:
            if rel == "../plugin/LoopSN2N_Fiji.jar":
                if jar:
                    plugin = HERE.parent / "plugin" / "LoopSN2N_Fiji.jar"
                    check(plugin, n, sha)
                    staged = stage / "LoopSN2N_Fiji.jar"
                    with plugin.open("rb") as f:
                        copy_stream(f, staged, n)
                    publish(staged, jar, n, sha)
            else:
                src = safe(HERE, rel)
                target = safe(dest, rel)
                # Do not hard-link repository files to the installation.
                if not target.exists():
                    staged = safe(stage, rel)
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    with src.open("rb") as f:
                        copy_stream(f, staged, n)
                    check(staged, n, sha)
                    publish(staged, target, n, sha)
        if biosr:
            for rel, n, sha, *_ in (MT, OTF, FAIR):
                target = safe(dest, rel)
                if not target.exists():
                    publish(safe(stage, rel), target, n, sha)
    print(f"Verified {a.example} example files: {dest}")
    if not a.files_only:
        python = env(dest, a.python)
        if a.python:
            print(f"Before starting Fiji, set LOOPSN2N_PYTHON to {python}")
    if not a.fiji:
        print(f"For a separate Fiji installation, set LOOPSN2N_EXAMPLES to {dest}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        sys.exit(f"Preparation stopped: {exc}")
