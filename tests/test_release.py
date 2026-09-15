from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[oprs]_[A-Za-z0-9_]{30,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def repository_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    )
    return [ROOT / line for line in output.splitlines() if line]


def test_private_runtime_files_are_not_tracked():
    relative = {path.relative_to(ROOT).as_posix() for path in repository_files()}

    assert not any(path == ".env" or path.startswith(".venv/") for path in relative)
    assert not any(path.endswith((".db", ".sqlite")) for path in relative)


def test_tracked_text_has_no_common_secret_shape_or_workspace_path():
    findings = []
    personal_root = f"{chr(47)}Users{chr(47)}"
    for path in repository_files():
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if personal_root in text:
            findings.append(f"personal path in {path.relative_to(ROOT)}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"secret-like value in {path.relative_to(ROOT)}")

    assert findings == []
