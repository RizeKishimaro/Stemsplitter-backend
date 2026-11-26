#!/usr/bin/env python3
"""Find a working Demucs Python or create DubPad's isolated runtime."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Callable, Sequence


IMPORT_DEMUCS_CHECK = "import demucs; print('DEMUSC_FOUND')"
WORKER_IMPORT_CHECK = "\n".join(
    (
        "import demucs",
        "import numpy",
        "import torch",
        "from demucs.api import Separator",
        "from demucs.audio import AudioFile",
        "import julius",
        "import soundfile",
        "from huggingface_hub import try_to_load_from_cache",
        "import yaml",
        "print('DEMUSC_WORKER_IMPORTS_FOUND')",
    )
)
MANAGED_MARKER = ".dubpad-managed-runtime.json"

Runner = Callable[..., subprocess.CompletedProcess[str]]


def candidate_interpreters(
    app_root: Path,
    *,
    home: Path | None = None,
    configured_python: str | None = None,
    runtime_dir: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
    executable: str | None = None,
) -> list[Path]:
    """Return configured, common venv, then PATH Python candidates."""
    home = (home or Path.home()).expanduser()
    values: list[Path] = []
    if configured_python:
        configured = Path(configured_python).expanduser()
        if configured.is_absolute() or "/" in configured_python:
            values.append(configured if configured.is_absolute() else app_root / configured)
        else:
            resolved = which(configured_python)
            if resolved:
                values.append(Path(resolved))

    values.extend(
        (
            home / ".venvs/demucs/bin/python",
            home / "venvs/demucs/bin/python",
            home / ".virtualenvs/demucs/bin/python",
            app_root / ".venv/bin/python",
            app_root / "venv/bin/python",
        )
    )
    if runtime_dir:
        values.append(runtime_dir / "bin/python")

    for name in ("python3", "python"):
        found = which(name)
        if found:
            values.append(Path(found))
    if executable:
        values.append(Path(executable))

    candidates: list[Path] = []
    seen: set[str] = set()
    for path in values:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            candidates.append(path)
    return candidates


def verify_interpreter(python: Path, run: Runner = subprocess.run) -> tuple[bool, str]:
    """Verify Demucs itself, then every non-stdlib worker import."""
    if not python.is_file():
        return False, "interpreter path does not exist"
    for code, marker, label in (
        (IMPORT_DEMUCS_CHECK, "DEMUSC_FOUND", "Demucs import"),
        (WORKER_IMPORT_CHECK, "DEMUSC_WORKER_IMPORTS_FOUND", "worker dependency imports"),
    ):
        try:
            result = run(
                [str(python), "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return False, f"{label} check failed: {error}"
        if result.returncode != 0 or marker not in result.stdout:
            detail = (result.stderr or result.stdout).strip()
            return False, f"{label} check failed" + (f": {detail}" if detail else "")
    return True, "Demucs and all worker imports are available"


def _venv_python(environment: Path) -> Path:
    return environment / "bin" / "python"


def ensure_runtime(
    app_root: Path,
    *,
    configured_python: str | None = None,
    runtime_dir: Path | None = None,
    home: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
    executable: str | None = None,
    run: Runner = subprocess.run,
    create_venv: Callable[[Path], None] | None = None,
) -> dict[str, str]:
    """Reuse a valid interpreter or install the pinned worker requirements."""
    app_root = app_root.expanduser().absolute()
    runtime_dir = runtime_dir or app_root / "data/runtime/demucs"
    if not runtime_dir.is_absolute():
        runtime_dir = app_root / runtime_dir
    report: list[str] = []

    candidates = candidate_interpreters(
        app_root,
        home=home,
        configured_python=configured_python,
        runtime_dir=runtime_dir,
        which=which,
        executable=executable,
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        valid, reason = verify_interpreter(candidate, run)
        if valid:
            report.append(f"Reusing compatible Demucs environment: {candidate}")
            return {"python": str(candidate), "action": "reused", "report": "\n".join(report)}
        report.append(f"Not compatible: {candidate} ({reason})")

    python = _venv_python(runtime_dir)
    marker = runtime_dir / MANAGED_MARKER
    runtime_is_managed = False
    if marker.is_file():
        try:
            runtime_is_managed = json.loads(marker.read_text(encoding="utf-8")).get("managedBy") == "DubPad"
        except (OSError, json.JSONDecodeError):
            runtime_is_managed = False
    if runtime_dir.exists() and not runtime_is_managed:
        raise RuntimeError(
            f"No compatible Demucs interpreter was found. The isolated runtime path "
            f"{runtime_dir} already exists but is not marked as DubPad-managed, so it "
            "was left untouched. Set DEMUCS_RUNTIME_DIR to another empty path."
        )

    if not runtime_is_managed:
        report.append(f"Creating isolated environment: {runtime_dir}")
        if create_venv:
            create_venv(runtime_dir)
        else:
            venv.EnvBuilder(with_pip=True, clear=False, symlinks=(os.name != "nt")).create(runtime_dir)
        marker.write_text(
            json.dumps({"managedBy": "DubPad", "status": "installing"}, indent=2) + "\n",
            encoding="utf-8",
        )

    requirements = app_root / "worker/requirements-demucs.txt"
    if not requirements.is_file():
        raise RuntimeError(f"Pinned Demucs requirements file is missing: {requirements}")
    install = run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "-r", str(requirements)],
        cwd=str(app_root),
        capture_output=True,
        text=True,
        timeout=60 * 60,
        check=False,
    )
    if install.returncode != 0:
        detail = (install.stderr or install.stdout).strip()
        raise RuntimeError(f"Could not install pinned Demucs worker dependencies: {detail}")

    valid, reason = verify_interpreter(python, run)
    if not valid:
        raise RuntimeError(
            f"Dependencies were installed into {runtime_dir}, but worker imports failed: {reason}. "
            "The persistent worker must report ready when the backend starts; model weights are "
            "never downloaded by this installer."
        )

    marker.write_text(
        json.dumps({"managedBy": "DubPad", "requirements": str(requirements)}, indent=2) + "\n",
        encoding="utf-8",
    )
    report.append(f"Installed and verified pinned worker dependencies in: {python}")
    return {"python": str(python), "action": "created", "report": "\n".join(report)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensure", action="store_true", help="reuse a valid Python or create the isolated runtime")
    parser.add_argument("--json", action="store_true", help="print a machine-readable result for the Rust backend")
    parser.add_argument("--app-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)

    if not args.ensure:
        parser.error("--ensure is required")
    try:
        result = ensure_runtime(
            args.app_root,
            configured_python=os.environ.get("DEMUCS_PYTHON"),
            runtime_dir=Path(os.environ.get("DEMUCS_RUNTIME_DIR", "data/runtime/demucs")),
            executable=sys.executable,
        )
    except Exception as error:
        print(f"Demucs runtime setup failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, separators=(",", ":")))
    else:
        print(result["report"])
        print(f"DEMUCS_PYTHON={result['python']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
