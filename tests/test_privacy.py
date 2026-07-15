from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".in",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
}
FORBIDDEN = {
    "exact_coordinates": re.compile("35" + r"\.856|83" + r"\.374"),
    "private_ipv4": re.compile("192" + r"\.168\.1\.223"),
    "street_address": re.compile("114" + r"7\s+Pine", re.IGNORECASE),
    "private_site_name": re.compile("Pine" + r"\s+Hollow", re.IGNORECASE),
    "private_city": re.compile("Sevier" + "ville", re.IGNORECASE),
    "private_state_context": re.compile("Tennes" + "see", re.IGNORECASE),
    "windows_private_path": re.compile(
        r"(?:C:\\\\|/mnt/c/)Users[/\\\\]" + "Jeff" + "rey", re.IGNORECASE
    ),
    "pi_private_path": re.compile("/home/" + "bird" + "netpi"),
    "hardcoded_password_helper": re.compile(
        r"echo\s+['\"]" + "bird" + "netpi" + r"['\"]", re.IGNORECASE
    ),
}


def tracked_text_files() -> list[Path]:
    return sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and not EXCLUDED_PARTS.intersection(p.parts)
        and p.suffix.lower() in TEXT_SUFFIXES
    )


def test_public_tree_contains_no_location_or_access_data() -> None:
    violations: list[str] = []
    for path in tracked_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                violations.append(f"{path.relative_to(ROOT)}:{label}")
    assert violations == []


def test_no_deployment_or_live_capture_scripts_are_published() -> None:
    forbidden_names = {"deploy.sh", "capture.sh", "insectnet_capture.py"}
    found = sorted(p.name for p in ROOT.rglob("*") if p.is_file() and p.name in forbidden_names)
    assert found == []
