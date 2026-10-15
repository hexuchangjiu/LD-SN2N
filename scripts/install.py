#!/usr/bin/env python3
"""Install the research Fiji plugin and minimal backend; preserve prior files in backups."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
import zipfile

REPOSITORY = Path(__file__).resolve().parents[1]
VERSION = "0.4.0"
JAR_NAME = "loopsn2n-calibrated-fiji-" + VERSION + ".jar"
COMMAND_CLASS = "org/loopsn2n/fiji/LoopSN2NCalibratedCommand.class"
PLUGIN_INDEX = "META-INF/json/org.scijava.plugin.Plugin"


def check_jar(path: Path, require_version: bool = False) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Plugin JAR must be a regular file: " + str(path))
    try:
        with zipfile.ZipFile(path) as archive:
            if COMMAND_CLASS not in archive.namelist() or PLUGIN_INDEX not in archive.namelist():
                raise ValueError("Not a calibrated LoopSN2N plugin: " + str(path))
            if require_version and ("Implementation-Version: " + VERSION) not in archive.read("META-INF/MANIFEST.MF").decode("utf-8"):
                raise ValueError("Expected a built " + VERSION + " plugin JAR")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("Invalid plugin JAR: " + str(path)) from exc


def is_link(path: Path) -> bool:
    # st_file_attributes also detects Windows junctions on Python 3.10/3.11,
    # where pathlib.Path.is_junction does not yet exist.
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def reject_links(root: Path) -> None:
    if is_link(root):
        raise ValueError("Symlink/junction paths are not supported for safe installation: " + str(root))
    if root.is_dir():
        for path in root.rglob("*"):
            if is_link(path):
                raise ValueError("Symlink/junction paths are not supported for safe installation: " + str(path))


def runtime_files(repository: Path) -> list[tuple[Path, Path]]:
    backend = repository / "fiji_calibrated_plugin/backend"
    if not (backend / "bridge.py").is_file():
        raise ValueError("The release backend is missing")
    reject_links(backend)
    selected = [(path, path.relative_to(repository)) for path in sorted(backend.rglob("*"))
                if path.is_file() and path.suffix in (".py", ".json") and "__pycache__" not in path.parts]
    checker = repository / "scripts/check_runtime.py"
    if not checker.is_file() or checker.is_symlink():
        raise ValueError("The release preflight scripts/check_runtime.py is missing")
    selected.append((checker, Path("scripts/check_runtime.py")))
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "LICENSE_STATUS.md",
                 "third_party/SN2N_upstream_LICENSE.txt", "requirements.txt"):
        path = repository / name
        if path.is_file() and not is_link(path):
            selected.append((path, Path(name)))
    return selected


def preflight(python: Path, checker: Path, temporary: Path, require_cuda: bool = False) -> dict:
    """Probe the chosen interpreter, not the interpreter executing this installer."""
    environment = os.environ.copy()
    temporary.mkdir(parents=True, exist_ok=True)
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", MPLCONFIGDIR=str(temporary / "matplotlib"),
                       XDG_CACHE_HOME=str(temporary / "cache"))
    for key in ("TEMP", "TMP", "TMPDIR"):
        environment[key] = str(temporary)
    command = [str(python), str(checker), "--json"]
    if require_cuda:
        command.append("--require-cuda")
    completed = subprocess.run(command, cwd=checker.parent.parent, env=environment, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=120)
    (temporary / "runtime_preflight.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (temporary / "runtime_preflight.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Runtime check failed (exit {completed.returncode}). See {temporary}; Fiji was not changed.")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Runtime checker did not return JSON; see " + str(temporary)) from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError("Runtime checker did not confirm readiness; see " + str(temporary))
    return payload


def install(fiji: Path, python: Path, jar: Path, temp_root: Path, require_cuda: bool = False,
            dry_run: bool = False, repository: Path = REPOSITORY) -> dict:
    # Check link identities before resolve() erases that information.
    for root in (fiji.expanduser(), fiji.expanduser() / "plugins", fiji.expanduser() / "jars"):
        if is_link(root):
            raise ValueError("Fiji root/plugins/jars must not be symlinks or junctions")
    fiji, python, jar = fiji.expanduser().resolve(), python.expanduser().resolve(), jar.expanduser().resolve()
    if not fiji.is_dir() or not (fiji / "jars").is_dir() or not (fiji / "plugins").is_dir():
        raise ValueError("--fiji must be an existing Fiji directory containing jars/ and plugins/")
    if not python.is_file():
        raise ValueError("--python must explicitly name an existing Python executable")
    check_jar(jar, require_version=True)
    files = runtime_files(repository)
    destination = fiji / "plugins/LoopSN2N"
    reject_links(destination)
    if destination.exists() and not destination.is_dir():
        raise ValueError("plugins/LoopSN2N is not a directory")
    old_jars = []
    for directory in (fiji / "jars", fiji / "plugins"):
        for path in sorted(directory.rglob("loopsn2n-calibrated-fiji-*.jar")):
            if destination not in path.parents:
                check_jar(path)
                if any(is_link(parent) for parent in path.parents if parent != fiji):
                    raise ValueError("Existing plugin JAR has a symlink parent: " + str(path))
                old_jars.append(path)
    for path in destination.glob("loopsn2n-calibrated-fiji-*.jar"):
        check_jar(path)
    temp_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="install-check-", dir=temp_root.resolve()))
    runtime_check = preflight(python, repository / "scripts/check_runtime.py", scratch, require_cuda)
    report = dict(status="DRY_RUN" if dry_run else "INSTALLED", version=VERSION,
                  destination=str(destination), python=str(python), runtime_preflight=runtime_check,
                  retired_calibrated_jars=[str(path) for path in old_jars],
                  runtime_files=len(files), scratch_directory=str(scratch),
                  note="Restart Fiji. Formal new-domain decisions remain DEFER; no GUI acceptance is claimed.")
    if dry_run:
        return report

    # Stage beside the target, ensuring renames are on the same filesystem.
    # The old directory is copied in full first, so unrelated files survive the upgrade.
    stage = Path(tempfile.mkdtemp(prefix=".loopsn2n-stage-", dir=fiji))
    backup = fiji / "LoopSN2N_backups" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    reject_links(fiji / "LoopSN2N_backups")
    moved = []
    previous_directory = False
    installed = False
    try:
        if destination.exists():
            shutil.copytree(destination, stage, dirs_exist_ok=True)
        # Only known calibrated JARs are retired; all are preserved in the original-directory backup.
        for path in stage.glob("loopsn2n-calibrated-fiji-*.jar"):
            path.unlink()
        shutil.copy2(jar, stage / JAR_NAME)
        for source, relative in files:
            target = stage / "runtime" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        configuration = dict(schema_version=1, release_version=VERSION, backend_root="runtime", python_executable=str(python))
        (stage / "runtime_config.json").write_text(json.dumps(configuration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report["backup_directory"] = str(backup)
        report["jar_sha256"] = hashlib.sha256((stage / JAR_NAME).read_bytes()).hexdigest()
        report["runtime_sha256"] = {relative.as_posix(): hashlib.sha256((stage / "runtime" / relative).read_bytes()).hexdigest()
                                    for _, relative in files}
        (stage / "installation_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        backup.mkdir(parents=True, exist_ok=False)
        if destination.exists():
            destination.rename(backup / "previous_LoopSN2N")
            previous_directory = True
        for old in old_jars:
            target = backup / "retired" / old.relative_to(fiji)
            target.parent.mkdir(parents=True, exist_ok=True)
            old.rename(target)
            moved.append((old, target))
        stage.rename(destination)
        installed = True
        (backup / "upgrade_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except BaseException:
        # Roll back every completed move; never delete original user content.
        if installed:
            destination.rename(stage)
        for old, saved in reversed(moved):
            saved.rename(old)
        if previous_directory:
            (backup / "previous_LoopSN2N").rename(destination)
        raise
    finally:
        # stage is an exact freshly-created child of Fiji, never a supplied/broad directory.
        if stage.exists():
            if stage.parent != fiji or not stage.name.startswith(".loopsn2n-stage-"):
                raise RuntimeError("Refusing to clean unexpected staging path")
            shutil.rmtree(stage)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiji", type=Path, required=True, help="Existing Fiji root; close Fiji before installing")
    parser.add_argument("--python", type=Path, required=True, help="Explicit Python executable with release dependencies")
    parser.add_argument("--jar", type=Path, default=REPOSITORY / "dist" / JAR_NAME)
    parser.add_argument("--temp-dir", type=Path, default=REPOSITORY / "build/install", help="Writable preflight cache/log directory")
    parser.add_argument("--require-cuda", action="store_true", help="Require working CUDA in addition to the CPU checks")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/runtime, but do not write into Fiji")
    args = parser.parse_args(argv)
    try:
        result = install(args.fiji, args.python, args.jar, args.temp_dir, args.require_cuda, args.dry_run)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, "Installation failed: " + str(exc) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
