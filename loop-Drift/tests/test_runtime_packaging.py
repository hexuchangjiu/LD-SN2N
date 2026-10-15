"""Single-JAR resource integrity and nested source-release packaging contracts."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts import build_fiji, install, package_release


class RuntimePackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="single-jar-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.module = self.root / "loop-Drift"
        self.contents = {
            "fiji_calibrated_plugin/backend/bridge.py": b"# bridge\n",
            "fiji_calibrated_plugin/backend/vendor/aunet.py": b"# unchanged network\n",
            "fiji_calibrated_plugin/backend/vendor/source_identity.json": b'{"source":"fixture"}\n',
            "scripts/check_runtime.py": b"# preflight\n",
            "requirements.txt": b"torch>=2.6\nnumpy\n",
            "LICENSE_STATUS.md": b"Author authorization pending\n",
            "third_party/SN2N_upstream_LICENSE.txt": b"Unchanged upstream notice\n",
        }
        for name, data in self.contents.items():
            target = self.module / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        self.classes = self.root / "classes"
        (self.classes / Path(install.COMMAND_CLASS).parent).mkdir(parents=True)
        (self.classes / install.COMMAND_CLASS).write_bytes(b"compiled fixture")
        (self.classes / Path(install.PLUGIN_INDEX).parent).mkdir(parents=True)
        (self.classes / install.PLUGIN_INDEX).write_bytes(b"fixture plugin index")

    def pack(self, name="plugin.jar"):
        output = self.root / name
        report = build_fiji.pack_jar(self.classes, output, install.VERSION, self.module)
        return output, report

    def test_bundle_exact_resources_checksums_and_reproducible_bytes(self):
        first, report = self.pack("first.jar")
        for path in self.module.rglob("*"):
            if path.is_file():
                os.utime(path, (1234567890, 1234567890))
        second, second_report = self.pack("second.jar")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(report, second_report)
        with zipfile.ZipFile(first) as jar:
            payload = install.validate_embedded_runtime(jar, install.VERSION)
            self.assertEqual(report["manifest_sha256"], hashlib.sha256(jar.read(install.RUNTIME_MANIFEST)).hexdigest())
            self.assertEqual([row["path"] for row in payload["files"]], sorted(self.contents))
            for row in payload["files"]:
                data = jar.read(install.RUNTIME_PREFIX + row["path"])
                self.assertEqual(data, self.contents[row["path"]])
                self.assertEqual(row["bytes"], len(data))
                self.assertEqual(row["sha256"], hashlib.sha256(data).hexdigest())
            self.assertTrue(all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in jar.infolist()))
        install.check_jar(first, require_version=True)

    def test_caches_python_dependencies_and_unrelated_assets_not_bundled(self):
        for name in ("fiji_calibrated_plugin/backend/__pycache__/ignored.py",
                     "fiji_calibrated_plugin/backend/.pytest_cache/example.json",
                     "fiji_calibrated_plugin/backend/model.pyc",
                     "fiji_calibrated_plugin/backend/model.pt",
                     ".venv/Lib/site-packages/torch/__init__.py",
                     "loopdrift/drifting.py"):
            target = self.module / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"must not be packaged")
        jar_path, _ = self.pack()
        with zipfile.ZipFile(jar_path) as jar:
            payload = install.validate_embedded_runtime(jar, install.VERSION)
            self.assertEqual({row["path"] for row in payload["files"]}, set(self.contents))

    def test_missing_required_notice_or_dependency_specification_fails(self):
        for name in ("requirements.txt", "LICENSE_STATUS.md", "third_party/SN2N_upstream_LICENSE.txt"):
            with self.subTest(name=name):
                target = self.module / name
                data = target.read_bytes()
                target.unlink()
                with self.assertRaises(ValueError):
                    self.pack()
                target.write_bytes(data)

    def test_repository_level_legal_notice_fallback(self):
        (self.module / "LICENSE_STATUS.md").rename(self.root / "LICENSE_STATUS.md")
        output, _ = self.pack()
        with zipfile.ZipFile(output) as jar:
            self.assertEqual(jar.read(install.RUNTIME_PREFIX + "LICENSE_STATUS.md"), self.contents["LICENSE_STATUS.md"])

    def test_unsafe_resource_paths_rejected_before_archive_creation(self):
        for name in ("../escape.py", "/absolute.py", "C:/escape.py", "safe/../escape.py",
                     "bad\\escape.py", "safe//file.py", "safe/./file.py", "CON", "file:stream", "trailing. "):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    install.safe_relative_name(name)
        source = self.module / "requirements.txt"
        with mock.patch.object(build_fiji, "runtime_files", return_value=[(source, Path("../escape.py"))]):
            with self.assertRaises(ValueError):
                self.pack()
        self.assertFalse((self.root / "plugin.jar").exists())

    def test_symlink_source_is_rejected(self):
        real_is_link = install.is_link
        bridge = self.module / "fiji_calibrated_plugin/backend/bridge.py"
        with mock.patch.object(install, "is_link", side_effect=lambda path: path == bridge or real_is_link(path)):
            with self.assertRaises(ValueError):
                self.pack()

    def test_corrupt_unlisted_duplicate_and_unsafe_manifest_entries_rejected(self):
        entries = build_fiji.runtime_entries(self.module, install.VERSION)
        mutations = []
        corrupted = dict(entries)
        corrupted[install.RUNTIME_PREFIX + "requirements.txt"] = b"tampered"
        mutations.append(list(corrupted.items()))
        unlisted = dict(entries)
        unlisted[install.RUNTIME_PREFIX + "unlisted.py"] = b"unknown"
        mutations.append(list(unlisted.items()))
        unsafe = dict(entries)
        manifest = json.loads(unsafe[install.RUNTIME_MANIFEST])
        manifest["files"][0]["path"] = "../escape.py"
        unsafe[install.RUNTIME_MANIFEST] = json.dumps(manifest).encode()
        mutations.append(list(unsafe.items()))
        manifest = json.loads(entries[install.RUNTIME_MANIFEST])
        manifest["files"].append(manifest["files"][0])
        duplicate = dict(entries)
        duplicate[install.RUNTIME_MANIFEST] = json.dumps(manifest).encode()
        mutations.append(list(duplicate.items()))
        for pairs in mutations:
            with self.subTest(resource_count=len(pairs)):
                stream = io.BytesIO()
                with zipfile.ZipFile(stream, "w") as jar:
                    for name, data in pairs:
                        jar.writestr(name, data)
                with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as jar, self.assertRaises(ValueError):
                    install.validate_embedded_runtime(jar, install.VERSION)


class SourcePackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nested-release-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.module = self.root / "loop-Drift"
        self.module.mkdir()
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True, capture_output=True)

    def add(self, name, data=b"fixture"):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        subprocess.run(["git", "add", "--", name], cwd=self.root, check=True, capture_output=True)
        return target

    def test_nested_script_uses_git_root_and_excludes_previous_releases_and_plugin(self):
        self.add("README.md")
        self.add("loop-Drift/scripts/build_fiji.py")
        self.add("loop-Drift/releases/0.4.0/old-source.zip")
        self.add("releases/previous.zip")
        self.add("plugin/LoopSN2N-0.5.0.jar")
        self.add("loop-Drift/dist/old.whl")
        root, files = package_release.source_files(self.module)
        self.assertEqual(root, self.root.resolve())
        self.assertEqual(set(files), {"README.md", "loop-Drift/scripts/build_fiji.py"})
        self.assertTrue(all(path.is_file() for path in files.values()))

    def test_research_asset_rejected(self):
        self.add("loop-Drift/secret.pt")
        with self.assertRaises(ValueError):
            package_release.source_files(self.module)

    def test_archive_is_reproducible_and_rejects_traversal(self):
        source = self.add("README.md")
        first, second = self.root / "one.zip", self.root / "two.zip"
        package_release.archive(first, {"README.md": source})
        package_release.archive(second, {"README.md": source})
        self.assertEqual(first.read_bytes(), second.read_bytes())
        unsafe = self.root / "unsafe.zip"
        with self.assertRaises(ValueError):
            package_release.archive(unsafe, {"../outside.md": source})
        self.assertFalse(unsafe.exists())


if __name__ == "__main__":
    unittest.main()
