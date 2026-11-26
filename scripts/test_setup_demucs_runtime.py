import subprocess
import tempfile
import unittest
from pathlib import Path

from setup_demucs_runtime import (
    IMPORT_DEMUCS_CHECK,
    MANAGED_MARKER,
    WORKER_IMPORT_CHECK,
    ensure_runtime,
)


class DemucsRuntimeSetupTests(unittest.TestCase):
    def test_reuses_configured_interpreter_after_verifying_worker_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preferred = root / "custom-env/bin/python"
            preferred.parent.mkdir(parents=True)
            preferred.touch()
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                if command[2] == IMPORT_DEMUCS_CHECK:
                    return subprocess.CompletedProcess(command, 0, "DEMUSC_FOUND\n", "")
                return subprocess.CompletedProcess(command, 0, "DEMUSC_WORKER_IMPORTS_FOUND\n", "")

            result = ensure_runtime(
                root,
                configured_python=str(preferred),
                home=root / "home",
                which=lambda _name: None,
                run=run,
            )

            self.assertEqual(result["action"], "reused")
            self.assertEqual(Path(result["python"]), preferred)
            self.assertIn("Reusing compatible Demucs environment", result["report"])
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][2], IMPORT_DEMUCS_CHECK)
            self.assertEqual(calls[1][2], WORKER_IMPORT_CHECK)
            self.assertFalse((root / "data/runtime/demucs").exists())

    def test_skips_demucs_only_environment_and_reuses_next_complete_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configured = root / "configured/bin/python"
            fallback = root / "home/.venvs/demucs/bin/python"
            configured.parent.mkdir(parents=True)
            fallback.parent.mkdir(parents=True)
            configured.touch()
            fallback.touch()

            def run(command, **kwargs):
                if command[0] == str(configured) and command[2] == WORKER_IMPORT_CHECK:
                    return subprocess.CompletedProcess(command, 1, "", "missing worker dependency")
                marker = "DEMUSC_FOUND" if command[2] == IMPORT_DEMUCS_CHECK else "DEMUSC_WORKER_IMPORTS_FOUND"
                return subprocess.CompletedProcess(command, 0, marker + "\n", "")

            result = ensure_runtime(
                root,
                configured_python=str(configured),
                home=root / "home",
                which=lambda _name: None,
                run=run,
            )

            self.assertEqual(result["action"], "reused")
            self.assertEqual(Path(result["python"]), fallback)
            self.assertIn("missing worker dependency", result["report"])
            self.assertFalse((root / "data/runtime/demucs").exists())

    def test_creates_isolated_runtime_only_when_no_candidate_is_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requirements = root / "worker/requirements-demucs.txt"
            requirements.parent.mkdir()
            requirements.write_text("demucs==4.1.0\n", encoding="utf-8")
            created = []

            def create_venv(path):
                created.append(path)
                (path / "bin").mkdir(parents=True)
                (path / "bin/python").touch()

            def run(command, **kwargs):
                if command[1:3] == ["-m", "pip"]:
                    return subprocess.CompletedProcess(command, 0, "installed\n", "")
                if command[2] == IMPORT_DEMUCS_CHECK:
                    return subprocess.CompletedProcess(command, 0, "DEMUSC_FOUND\n", "")
                return subprocess.CompletedProcess(command, 0, "DEMUSC_WORKER_IMPORTS_FOUND\n", "")

            result = ensure_runtime(
                root,
                runtime_dir=Path("data/runtime/demucs"),
                home=root / "home",
                which=lambda _name: None,
                run=run,
                create_venv=create_venv,
            )

            runtime = root / "data/runtime/demucs"
            self.assertEqual(result["action"], "created")
            self.assertEqual(Path(result["python"]), runtime / "bin/python")
            self.assertEqual(created, [runtime])
            self.assertTrue((runtime / MANAGED_MARKER).is_file())
            self.assertIn("Installed and verified", result["report"])

    def test_does_not_modify_an_existing_unmanaged_runtime_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir()
            sentinel = runtime / "keep.txt"
            sentinel.write_text("untouched", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "left untouched"):
                ensure_runtime(
                    root,
                    runtime_dir=runtime,
                    home=root / "home",
                    which=lambda _name: None,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched")


if __name__ == "__main__":
    unittest.main()
