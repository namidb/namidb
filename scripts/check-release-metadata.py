"""Validate NamiDB release metadata before an immutable tag is published."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        help="Release tag to validate, for example v2.0.1 or py-v2.0.1",
    )
    parser.add_argument(
        "--tag-kind",
        choices=("engine", "python"),
        help="Expected tag family; required when --tag is supplied",
    )
    args = parser.parse_args()

    if bool(args.tag) != bool(args.tag_kind):
        parser.error("--tag and --tag-kind must be supplied together")

    errors: list[str] = []
    root_manifest = load_toml(REPO_ROOT / "Cargo.toml")
    workspace = root_manifest["workspace"]
    version = str(workspace["package"]["version"])

    member_names: set[str] = set()
    for member in workspace["members"]:
        manifest_path = REPO_ROOT / member / "Cargo.toml"
        manifest = load_toml(manifest_path)
        package = manifest["package"]
        name = str(package["name"])
        member_names.add(name)
        declared = package["version"]
        if isinstance(declared, dict) and declared.get("workspace") is True:
            resolved = version
        else:
            resolved = str(declared)
        if resolved != version:
            errors.append(
                f"{manifest_path.relative_to(REPO_ROOT)}: package {name!r} "
                f"resolves to {resolved}, expected {version}"
            )

    for name, dependency in workspace.get("dependencies", {}).items():
        if not isinstance(dependency, dict) or "path" not in dependency:
            continue
        pinned = dependency.get("version")
        if pinned is None:
            errors.append(
                f"Cargo.toml: local dependency {name!r} has no publishable version pin"
            )
        elif str(pinned) != version:
            errors.append(
                f"Cargo.toml: local dependency {name!r} pins {pinned}, "
                f"expected {version}"
            )

    pyproject_path = REPO_ROOT / "crates/namidb-py/pyproject.toml"
    pyproject = load_toml(pyproject_path)
    python_version = str(pyproject["project"]["version"])
    if python_version != version:
        errors.append(
            f"{pyproject_path.relative_to(REPO_ROOT)}: project.version is "
            f"{python_version}, expected {version}"
        )

    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release_heading = re.compile(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}(?:\s|:|$)",
        re.MULTILINE,
    )
    if not release_heading.search(changelog):
        errors.append(
            f"CHANGELOG.md: no dated release heading exists for version {version}"
        )

    lockfile = load_toml(REPO_ROOT / "Cargo.lock")
    locked_members = {
        str(package["name"]): str(package["version"])
        for package in lockfile["package"]
        if str(package["name"]) in member_names
    }
    for name in sorted(member_names):
        locked = locked_members.get(name)
        if locked is None:
            errors.append(f"Cargo.lock: workspace package {name!r} is missing")
        elif locked != version:
            errors.append(
                f"Cargo.lock: workspace package {name!r} is {locked}, "
                f"expected {version}"
            )

    canonical_license = REPO_ROOT / "LICENSE"
    python_license = REPO_ROOT / "crates/namidb-py/LICENSE"
    if not python_license.exists():
        errors.append("crates/namidb-py/LICENSE is missing")
    elif python_license.read_bytes() != canonical_license.read_bytes():
        errors.append("crates/namidb-py/LICENSE differs from the repository LICENSE")

    project_license = pyproject["project"].get("license")
    if project_license != "BUSL-1.1":
        errors.append(
            "crates/namidb-py/pyproject.toml: project.license must be "
            'the SPDX expression "BUSL-1.1"'
        )
    if "LICENSE" not in pyproject["project"].get("license-files", []):
        errors.append(
            "crates/namidb-py/pyproject.toml: project.license-files must "
            'include "LICENSE"'
        )

    if args.tag:
        prefix = "v" if args.tag_kind == "engine" else "py-v"
        expected_tag = f"{prefix}{version}"
        if args.tag != expected_tag:
            errors.append(
                f"release tag is {args.tag!r}, expected {expected_tag!r} "
                f"from declared version {version}"
            )

    if errors:
        for error in errors:
            print(f"release metadata error: {error}", file=sys.stderr)
        return 1

    tag_note = f", tag={args.tag}" if args.tag else ""
    print(f"release metadata ok: version={version}{tag_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
