"""Portable installer/build regression checks; all Fiji installations are disposable mocks."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts import build_fiji, install


class InstallBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="loopsn2n-install-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = self.root / "release 中文 空格"
        self.fiji = self.root / "Fiji Space.app"
        (self.fiji / "jars").mkdir(parents=True)
        (self.fiji / "plugins").mkdir()
        backend = self.repository / "fiji_calibrated_plugin/backend"
        backend.mkdir(parents=True)
        (backend / "bridge.py").write_text("# fixture, never executed\n", encoding="utf-8")
        (backend / "vendor").mkdir()
        (backend / "vendor/source_identity.json").write_text("{}", encoding="utf-8")
        (backend / "__pycache__").mkdir()
        (backend / "__pycache__/ignored.py").write_text("# cache", encoding="utf-8")
        (self.repository / "scripts").mkdir()
        (self.repository / "scripts/check_runtime.py").write_text("print('{}')", encoding="utf-8")
        (self.repository / "LICENSE_STATUS.md").write_text("Private release; permission pending", encoding="utf-8")
        (self.repository / "third_party").mkdir()
        (self.repository / "third_party/SN2N_upstream_LICENSE.txt").write_text("Preserve upstream notice", encoding="utf-8")
        self.jar = self.repository / install.JAR_NAME
        self.make_jar(self.jar)
        self.target = self.fiji / "plugins/LoopSN2N"
        self.cache = self.root / "scratch"
        self.preflight = mock.patch.object(install, "preflight", return_value={"status": "PASS"})
        self.preflight_mock = self.preflight.start()
        self.addCleanup(self.preflight.stop)

    def make_jar(self, path, version="0.4.0"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as jar:
            jar.writestr(install.COMMAND_CLASS, b"fixture")
            jar.writestr(install.PLUGIN_INDEX, b"fixture")
            jar.writestr("META-INF/MANIFEST.MF", "Implementation-Version: " + version + "\r\n")

    def do_install(self, **options):
        return install.install(self.fiji, Path(sys.executable), self.jar, self.cache,
                               repository=self.repository, **options)

    def test_fresh_install_has_relative_runtime_and_explicit_python(self):
        result = self.do_install()
        config = json.loads((self.target / "runtime_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["backend_root"], "runtime")
        self.assertEqual(config["python_executable"], str(Path(sys.executable).resolve()))
        self.assertTrue((self.target / "runtime/fiji_calibrated_plugin/backend/bridge.py").is_file())
        self.assertTrue((self.target / "runtime/scripts/check_runtime.py").is_file())
        self.assertTrue((self.target / "runtime/LICENSE_STATUS.md").is_file())
        self.assertEqual((self.target / "runtime/third_party/SN2N_upstream_LICENSE.txt").read_text(encoding="utf-8"),
                         "Preserve upstream notice")
        self.assertFalse((self.target / "runtime/fiji_calibrated_plugin/backend/__pycache__").exists())
        self.assertEqual(result["status"], "INSTALLED")
        self.preflight_mock.assert_called_once()

    def test_upgrade_preserves_unrelated_files_and_backs_up_old_plugin(self):
        self.target.mkdir()
        note = self.target / "my-notes.txt"
        note.write_text("keep me", encoding="utf-8")
        old = self.target / "loopsn2n-calibrated-fiji-0.3.0.jar"
        self.make_jar(old, version="0.3.0")
        result = self.do_install()
        self.assertEqual(note.read_text(encoding="utf-8"), "keep me")
        backup = Path(result["backup_directory"]) / "previous_LoopSN2N"
        self.assertTrue((backup / old.name).is_file())
        self.assertEqual((backup / note.name).read_text(encoding="utf-8"), "keep me")
        self.assertFalse(old.exists())
        self.assertTrue((self.target / install.JAR_NAME).is_file())

    def test_known_older_calibrated_jars_retired_without_touching_strict_plugin(self):
        old = self.fiji / "jars/loopsn2n-calibrated-fiji-0.2.0.jar"
        self.make_jar(old, version="0.2.0")
        strict = self.fiji / "jars/loopsn2n-fiji-0.1.0-SNAPSHOT.jar"
        strict.write_bytes(b"leave the independent strict plugin intact")
        result = self.do_install()
        self.assertFalse(old.exists())
        self.assertTrue((Path(result["backup_directory"]) / "retired/jars" / old.name).is_file())
        self.assertEqual(strict.read_bytes(), b"leave the independent strict plugin intact")

    def test_misleading_jar_name_is_rejected_not_removed(self):
        foreign = self.fiji / "jars/loopsn2n-calibrated-fiji-foreign.jar"
        foreign.write_bytes(b"not ours")
        with self.assertRaises(ValueError):
            self.do_install()
        self.assertEqual(foreign.read_bytes(), b"not ours")
        self.assertFalse(self.target.exists())

    def test_preflight_failure_does_not_modify_fiji(self):
        self.preflight_mock.side_effect = RuntimeError("missing torch")
        with self.assertRaises(RuntimeError):
            self.do_install()
        self.assertEqual(sorted(path.name for path in self.fiji.iterdir()), ["jars", "plugins"])
        self.assertFalse(self.target.exists())

    def test_dry_run_changes_no_fiji_files(self):
        result = self.do_install(dry_run=True)
        self.assertEqual(result["status"], "DRY_RUN")
        self.assertEqual(sorted(path.name for path in self.fiji.iterdir()), ["jars", "plugins"])
        self.assertFalse(self.target.exists())

    def test_rollback_restores_all_prior_files_after_publish_failure(self):
        self.target.mkdir()
        note = self.target / "prior.txt"
        note.write_text("original", encoding="utf-8")
        old = self.fiji / "jars/loopsn2n-calibrated-fiji-0.3.0.jar"
        self.make_jar(old, version="0.3.0")
        original_rename = Path.rename

        def fail_publish(path, destination):
            if path.name.startswith(".loopsn2n-stage-") and Path(destination) == self.target:
                raise OSError("simulated locked target")
            return original_rename(path, destination)

        with mock.patch.object(Path, "rename", fail_publish), self.assertRaises(OSError):
            self.do_install()
        self.assertEqual(note.read_text(encoding="utf-8"), "original")
        self.assertTrue(old.is_file())
        self.assertFalse((self.target / install.JAR_NAME).exists())
        self.assertEqual(list(self.fiji.glob(".loopsn2n-stage-*")), [])

    def test_wrong_release_version_rejected_before_preflight(self):
        self.make_jar(self.jar, version="0.3.0")
        with self.assertRaises(ValueError):
            self.do_install()
        self.preflight_mock.assert_not_called()

    def test_runtime_missing_checker_is_rejected(self):
        (self.repository / "scripts/check_runtime.py").unlink()
        with self.assertRaises(ValueError):
            self.do_install()
        self.assertFalse(self.target.exists())

    def test_archive_is_deterministic_with_version_manifest(self):
        classes = self.root / "classes"
        classes.mkdir()
        (classes / "Example.class").write_bytes(b"test class")
        first, second = self.root / "first.jar", self.root / "second.jar"
        build_fiji.pack_jar(classes, first, "0.4.0")
        build_fiji.pack_jar(classes, second, "0.4.0")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as jar:
            self.assertIn("Implementation-Version: 0.4.0", jar.read("META-INF/MANIFEST.MF").decode("utf-8"))

    def test_argument_file_quotes_spaces_unicode_and_windows_paths(self):
        args = self.root / "arguments.txt"
        build_fiji.argument_file(args, ["C:\\Program Files\\Fiji", "中文 空格", "hash#name"])
        self.assertEqual(args.read_text(encoding="utf-8").splitlines(),
                         ['"C:\\\\Program Files\\\\Fiji"', '"中文 空格"', '"hash#name"'])
        with self.assertRaises(ValueError):
            build_fiji.argument_file(args, ["bad\nargument"])

    def test_explicit_missing_jdk_does_not_fall_back_silently(self):
        with self.assertRaises(ValueError):
            build_fiji.find_jdk(self.fiji, self.root / "missing_jdk")

    def test_manifest_classpath_preserves_unicode_spaces_and_line_limit(self):
        output = self.root / "classpath.jar"
        build_fiji.classpath_jar(output, [self.repository, self.jar])
        with zipfile.ZipFile(output) as archive:
            raw = archive.read("META-INF/MANIFEST.MF").decode("ascii")
        self.assertTrue(all(len(line) <= 70 for line in raw.splitlines()))
        unfolded = raw.replace("\r\n ", "")
        self.assertIn(self.repository.as_uri() + "/", unfolded)
        self.assertIn(self.jar.as_uri(), unfolded)

    def test_installer_and_frontend_versions_match(self):
        self.assertEqual(install.VERSION, build_fiji.version())


if __name__ == "__main__":
    unittest.main()
