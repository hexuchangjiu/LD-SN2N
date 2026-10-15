#!/usr/bin/env python3
"""Build the Fiji frontend using an existing Fiji and JDK (no Maven or downloads)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

try:
    from .install import (RUNTIME_MANIFEST, RUNTIME_PREFIX, reject_links, runtime_files,
                          safe_relative_name, validate_embedded_runtime)
except ImportError:  # Direct execution: python scripts/build_fiji.py.
    from install import (RUNTIME_MANIFEST, RUNTIME_PREFIX, reject_links, runtime_files,
                         safe_relative_name, validate_embedded_runtime)

REPOSITORY = Path(__file__).resolve().parents[1]
MAIN_CLASS = "org.loopsn2n.fiji.LoopSN2NCalibratedCommand"
PLUGIN_INDEX = "META-INF/json/org.scijava.plugin.Plugin"


def version(repository: Path = REPOSITORY) -> str:
    source = repository / "fiji_calibrated_plugin/java/src/main/java/org/loopsn2n/fiji/LoopSN2NCalibratedCommand.java"
    match = re.search(r'public static final String VERSION = "([0-9.]+)";', source.read_text(encoding="utf-8"))
    if not match:
        raise ValueError("Frontend version is missing")
    return match.group(1)


def find_jdk(fiji: Path, supplied: Path | None) -> Path:
    """Prefer the explicitly selected JDK, then Fiji's bundled JDK, then JAVA_HOME/PATH."""
    executable = "javac.exe" if os.name == "nt" else "javac"
    candidates = []
    if supplied is not None:
        candidates = [supplied / "bin", supplied]
    else:
        bundled = fiji / "java"
        if bundled.is_dir():
            candidates.extend(path.parent for path in sorted(bundled.rglob(executable)))
        if os.environ.get("JAVA_HOME"):
            candidates.append(Path(os.environ["JAVA_HOME"]) / "bin")
        located = shutil.which(executable)
        if located:
            candidates.append(Path(located).parent)
    java = "java.exe" if os.name == "nt" else "java"
    for candidate in candidates:
        if (candidate / executable).is_file() and (candidate / java).is_file():
            return candidate.resolve()
    raise ValueError("A JDK 11+ is required. Supply --jdk <JDK-home-or-bin>; a JRE alone cannot compile.")


def argument_file(path: Path, arguments: list[str]) -> None:
    # javac argument files avoid the Windows command-length limit. The Java
    # launcher uses native.encoding (not file.encoding) on Windows, so runtime
    # launches instead use the manifest classpath below and direct Unicode argv.
    # Double-quoted entries retain spaces, Unicode and '#' in filesystem paths.
    def quote(value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("Newlines in build arguments are unsupported")
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    path.write_text("\n".join(quote(str(value)) for value in arguments) + "\n", encoding="utf-8")


def classpath_jar(output: Path, entries: list[Path]) -> Path:
    """A short ASCII manifest replaces a potentially huge or Unicode runtime classpath."""
    urls = [path.resolve().as_uri() + ("/" if path.is_dir() else "") for path in entries]
    line = "Class-Path: " + " ".join(urls)
    # Manifest continuation lines must begin with one space; URLs are ASCII escaped.
    folded = [line[:70]]
    line = line[70:]
    while line:
        folded.append(" " + line[:69])
        line = line[69:]
    manifest = "Manifest-Version: 1.0\r\n" + "\r\n".join(folded) + "\r\n\r\n"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("META-INF/MANIFEST.MF", manifest.encode("ascii"))
    return output


def runtime_entries(repository: Path, release_version: str) -> dict[str, bytes]:
    """Freeze the minimal Python programs and notices, never interpreter/dependency binaries."""
    entries = {}
    records = []
    for source, relative in runtime_files(repository):
        name = safe_relative_name(relative.as_posix())
        data = source.read_bytes()
        entries[RUNTIME_PREFIX + name] = data
        records.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    manifest = dict(version=release_version, files=records)
    entries[RUNTIME_MANIFEST] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")) + "\n").encode("utf-8")
    return entries


def pack_jar(classes: Path, output: Path, release_version: str,
             repository: Path = REPOSITORY) -> dict:
    """Build and validate a deterministic single-file plugin before atomic publication."""
    reject_links(classes)
    entries = {"META-INF/MANIFEST.MF": ("Manifest-Version: 1.0\r\n"
               "Implementation-Title: LoopSN2N calibrated Fiji\r\n"
               f"Implementation-Version: {release_version}\r\n\r\n").encode("utf-8")}
    for path in sorted(classes.rglob("*")):
        if path.is_file():
            name = safe_relative_name(path.relative_to(classes).as_posix())
            if name in entries or name.startswith(RUNTIME_PREFIX):
                raise ValueError("Reserved JAR resource path in compiled classes: " + name)
            entries[name] = path.read_bytes()
    entries.update(runtime_entries(repository, release_version))
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=".loopsn2n-", suffix=".jar", dir=output.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as jar:
            for name, data in sorted(entries.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o644 << 16
                jar.writestr(info, data)
        with zipfile.ZipFile(temporary) as verify:
            if verify.testzip() is not None:
                raise ValueError("JAR archive integrity check failed")
            manifest = validate_embedded_runtime(verify, release_version)
        os.replace(temporary, output)
        return dict(manifest_sha256=hashlib.sha256(entries[RUNTIME_MANIFEST]).hexdigest(),
                    files=len(manifest["files"]), bytes=sum(row["bytes"] for row in manifest["files"]))
    finally:
        if temporary.exists():
            temporary.unlink()


def build(fiji: Path, jdk: Path | None, output: Path, work_dir: Path,
          run_smoke: bool = True, repository: Path = REPOSITORY) -> dict:
    fiji = fiji.expanduser().resolve()
    if not (fiji / "jars").is_dir():
        raise ValueError("--fiji must be the Fiji directory containing jars/ (on macOS: Fiji.app)")
    jdk_bin = find_jdk(fiji, jdk)
    suffix = ".exe" if os.name == "nt" else ""
    javac, java = jdk_bin / ("javac" + suffix), jdk_bin / ("java" + suffix)
    dependencies = sorted(path for path in (fiji / "jars").rglob("*.jar")
                          if not re.match(r"(?i)loopsn2n-(?:calibrated-fiji-|[0-9])", path.name))
    if not dependencies:
        raise ValueError("No Fiji dependency JARs found")
    classpath = os.pathsep.join(str(path) for path in dependencies)
    work_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="build-", dir=work_dir.resolve()))
    classes, tests = scratch / "classes", scratch / "test_classes"
    classes.mkdir()
    tests.mkdir()
    temporary = scratch / "tmp"
    temporary.mkdir()
    environment = os.environ.copy()
    for key in ("TEMP", "TMP", "TMPDIR"):
        environment[key] = str(temporary)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    def run(label: str, executable: Path, arguments: list[str], javac_command: bool = False) -> str:
        prefix = [str(executable)]
        if javac_command:
            arguments_path = scratch / (label + ".args")
            argument_file(arguments_path, arguments)
            prefix += ["-J-Dfile.encoding=UTF-8", "-J-Djava.io.tmpdir=" + str(temporary)]
            arguments = ["@" + str(arguments_path)]
        result = subprocess.run(prefix + arguments, cwd=repository, env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding="utf-8", errors="replace", timeout=180)
        (scratch / (label + ".log")).write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{label} failed (exit {result.returncode}); see {scratch / (label + '.log')}")
        return result.stdout

    compiler = subprocess.run([str(javac), "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              encoding="utf-8", errors="replace", timeout=30, env=environment)
    match = re.search(r"javac (\d+)", compiler.stdout)
    if compiler.returncode or not match or int(match.group(1)) < 11:
        raise ValueError("This builder requires JDK 11+; the resulting plugin targets Java 8")
    source_root = repository / "fiji_calibrated_plugin/java/src"
    sources = sorted((source_root / "main/java").rglob("*.java"))
    test_sources = sorted((source_root / "test/java").rglob("*.java"))
    if not sources or not test_sources:
        raise ValueError("Complete Java main and test sources are required")
    run("compile", javac, ["--release", "8", "-encoding", "UTF-8", "-classpath", classpath,
                          "-processor", "org.scijava.annotations.AnnotationProcessor", "-d", str(classes)]
        + [str(path) for path in sources], javac_command=True)
    index = classes / PLUGIN_INDEX
    if not index.is_file() or MAIN_CLASS not in index.read_text(encoding="utf-8"):
        raise RuntimeError("The generated SciJava plugin index does not contain the research command")
    output = output.resolve()
    release_version = version(repository)
    # Verify the staged JAR before replacing any previously successful build.
    staged_jar = scratch / ("LoopSN2N-" + release_version + ".jar")
    runtime = pack_jar(classes, staged_jar, release_version, repository)
    run("compile_tests", javac, ["--release", "8", "-encoding", "UTF-8", "-proc:none", "-classpath",
                                 str(classes) + os.pathsep + classpath, "-d", str(tests)]
        + [str(path) for path in test_sources], javac_command=True)
    smoke_results = {}
    if run_smoke:
        runtime_cp = str(classpath_jar(scratch / "smoke-classpath.jar", [tests, staged_jar] + dependencies))
        for name, extra in [("CalibratedFrontendSmokeTest", [str(scratch / "frontend")]),
                            ("CalibratedRuntimeSmokeTest", [str(scratch / "runtime")]),
                            ("CalibratedSingleJarSmokeTest", [str(scratch / "singlejar")]),
                            ("CalibratedPluginDiscoverySmokeTest", [])]:
            smoke_results[name] = run(name, java, ["-Djava.awt.headless=true", "-Dfile.encoding=UTF-8",
                "-Djava.io.tmpdir=" + str(temporary), "--add-opens=java.base/java.lang=ALL-UNNAMED",
                "-classpath", runtime_cp, "org.loopsn2n.fiji." + name] + extra)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, pending_name = tempfile.mkstemp(prefix=".loopsn2n-tested-", suffix=".jar", dir=output.parent)
    os.close(handle)
    pending = Path(pending_name)
    try:
        # Publish the exact bytes exercised by the smoke tests, not a second
        # assembly that could pick up concurrent source edits.
        shutil.copyfile(staged_jar, pending)
        if pending.read_bytes() != staged_jar.read_bytes():
            raise RuntimeError("The tested plugin could not be copied without changes")
        os.replace(pending, output)
    finally:
        if pending.exists():
            pending.unlink()
    result = dict(status="PASS", version=release_version, jar=str(output),
                  sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  compiler=compiler.stdout.strip(), dependency_jars=len(dependencies),
                  work_directory=str(scratch), test_classes=str(tests), java=str(java),
                  smoke_tests=smoke_results, gui_interactive_verified=False)
    result["embedded_runtime"] = runtime
    result["single_jar"] = True
    result["python_dependencies_bundled"] = False
    (scratch / "build_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiji", type=Path, required=True, help="Installed Fiji directory (read-only)")
    parser.add_argument("--jdk", type=Path, help="JDK 11+ home or bin; defaults to the Fiji bundled JDK")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "dist" / ("LoopSN2N-" + version() + ".jar"))
    parser.add_argument("--work-dir", type=Path, default=REPOSITORY / "build/fiji", help="Writable build/cache/log directory")
    parser.add_argument("--skip-smoke", action="store_true", help="Compile only; interactive GUI is never claimed verified")
    args = parser.parse_args(argv)
    try:
        result = build(args.fiji, args.jdk, args.output, args.work_dir, not args.skip_smoke)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, "Build failed: " + str(exc) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
