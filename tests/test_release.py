from __future__ import annotations

import json
import re
import subprocess
import sys
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


def test_static_analysis_deck_is_current_and_bounded():
    result = subprocess.run(
        [sys.executable, "scripts/export_static_analyses.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads((ROOT / "docs/demo/analyses.json").read_text())

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(payload["analyses"]) == 5
    assert {analysis["ticker"] for analysis in payload["analyses"]} == {
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "TSLA",
    }
    assert all(len(analysis["series"]) <= 64 for analysis in payload["analyses"])


def test_static_demo_has_no_runtime_third_party_assets():
    html = (ROOT / "docs/demo/index.html").read_text()

    assert 'src="http' not in html
    assert 'href="http' not in html.replace(
        'href="https://github.com/gauravsdama/deepTradeAlgo"',
        "",
    )
