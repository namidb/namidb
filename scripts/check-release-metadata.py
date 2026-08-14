"""Validate NamiDB release metadata before an immutable tag is published."""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(
    r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-(?:"
    r"(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*"
    r"))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
ACTION_USE = re.compile(
    r"^\s*(?:-\s*)?uses:\s*(?P<target>\S+)"
    r"(?:\s+#\s*(?P<version>.+?))?\s*$"
)
PINNED_ACTION = re.compile(
    r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*"
    r"@[0-9a-f]{40}$"
)
VERSION_COMMENT = re.compile(r"^(?:v?\d|stable(?:\s|$)|release/v\d)")
FROM_IMAGE = re.compile(
    r"^\s*FROM\s+(?:--platform=\S+\s+)?(?P<image>\S+)",
    re.IGNORECASE,
)
PINNED_IMAGE = re.compile(
    r"^(?P<tagged>[^@\s]+)@sha256:(?P<digest>[0-9a-f]{64})$"
)


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def validate_workflow_action_pins(errors: list[str]) -> None:
    """Reject mutable third-party action refs before they reach a runner."""

    workflow_root = REPO_ROOT / ".github/workflows"
    workflows = set(workflow_root.glob("*.yml"))
    workflows.update(workflow_root.glob("*.yaml"))
    for workflow in sorted(workflows):
        lines = workflow.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            match = ACTION_USE.match(line)
            if not match:
                continue
            target = match.group("target")
            if target.startswith("./"):
                continue
            location = f"{workflow.relative_to(REPO_ROOT)}:{line_number}"
            if not PINNED_ACTION.fullmatch(target):
                errors.append(
                    f"{location}: external action {target!r} must use a full "
                    "40-character commit SHA"
                )
            version_comment = (match.group("version") or "").strip()
            if not VERSION_COMMENT.match(version_comment):
                errors.append(
                    f"{location}: pinned action must retain a human-readable "
                    "version comment (for example '# v4.2.0')"
                )
            if target.startswith("actions/checkout@"):
                step_indent = len(line) - len(line.lstrip())
                checkout_body: list[str] = []
                for following in lines[line_number:]:
                    following_indent = len(following) - len(following.lstrip())
                    if (
                        following.lstrip().startswith("- ")
                        and following_indent <= step_indent
                    ):
                        break
                    checkout_body.append(following)
                if not any(
                    re.fullmatch(r"\s*persist-credentials:\s*false\s*", body_line)
                    for body_line in checkout_body
                ):
                    errors.append(
                        f"{location}: checkout must set "
                        "persist-credentials: false"
                    )


def validate_docker_base_pins(errors: list[str], rust_version: str) -> None:
    """Require tagged, multi-arch-friendly digest pins for every base image."""

    dockerfile = REPO_ROOT / "crates/namidb-server/Dockerfile"
    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    syntax = re.search(r"^#\s*syntax=(?P<image>\S+)\s*$", dockerfile_text, re.MULTILINE)
    if syntax is None or not PINNED_IMAGE.fullmatch(syntax.group("image")):
        errors.append(
            "crates/namidb-server/Dockerfile: syntax frontend must retain a "
            "tag and immutable @sha256 digest"
        )

    rust_base_version: str | None = None
    for line_number, line in enumerate(
        dockerfile_text.splitlines(),
        start=1,
    ):
        match = FROM_IMAGE.match(line)
        if not match:
            continue
        image = match.group("image")
        if image.lower() == "scratch":
            continue
        pinned = PINNED_IMAGE.fullmatch(image)
        location = f"{dockerfile.relative_to(REPO_ROOT)}:{line_number}"
        if not pinned:
            errors.append(
                f"{location}: base image {image!r} must include an immutable "
                "@sha256 manifest-list digest"
            )
            continue
        tagged = pinned.group("tagged")
        final_component = tagged.rsplit("/", 1)[-1]
        if ":" not in final_component:
            errors.append(
                f"{location}: digest-pinned base image must retain a "
                "human-readable tag"
            )
        rust_match = re.fullmatch(
            r"rust:(?P<version>\d+\.\d+\.\d+)-.+",
            final_component,
        )
        if rust_match:
            rust_base_version = rust_match.group("version")

    if rust_base_version is None:
        errors.append(
            "crates/namidb-server/Dockerfile: no pinned rust:<version> "
            "builder image found"
        )
        return
    expected_minor = ".".join(rust_version.split(".")[:2])
    actual_minor = ".".join(rust_base_version.split(".")[:2])
    if actual_minor != expected_minor:
        errors.append(
            "crates/namidb-server/Dockerfile: Rust builder "
            f"{rust_base_version} does not match workspace rust-version "
            f"{rust_version}"
        )


def validate_documented_image_major(errors: list[str], version: str) -> None:
    expected_major = version.split(".", 1)[0]
    paths = (
        REPO_ROOT / "docker-compose.yml",
        REPO_ROOT / "README.md",
        REPO_ROOT / "crates/namidb-server/README.md",
    )
    image_tag = re.compile(r"\bnamidb/namidb-server:(?P<major>\d+)\b")
    for path in paths:
        majors = {
            match.group("major")
            for match in image_tag.finditer(path.read_text(encoding="utf-8"))
        }
        if not majors:
            errors.append(
                f"{path.relative_to(REPO_ROOT)}: no documented "
                "namidb/namidb-server:<major> image reference"
            )
        elif majors != {expected_major}:
            errors.append(
                f"{path.relative_to(REPO_ROOT)}: image major aliases "
                f"{sorted(majors)} do not match release major {expected_major}"
            )


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
    rust_version = str(workspace["package"]["rust-version"])
    if not SEMVER.fullmatch(version):
        errors.append(
            f"Cargo.toml: workspace package version {version!r} is not strict SemVer"
        )

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
        rf"^## \[{re.escape(version)}\] - "
        rf"(?P<date>\d{{4}}-\d{{2}}-\d{{2}})(?:\s|:|$)",
        re.MULTILINE,
    )
    release_headings = list(release_heading.finditer(changelog))
    if not release_headings:
        errors.append(
            f"CHANGELOG.md: no dated release heading exists for version {version}"
        )
    elif len(release_headings) != 1:
        errors.append(
            f"CHANGELOG.md: release {version} has {len(release_headings)} "
            "dated headings; expected exactly one"
        )
    else:
        try:
            datetime.date.fromisoformat(release_headings[0].group("date"))
        except ValueError:
            errors.append(
                f"CHANGELOG.md: release {version} has an invalid calendar date"
            )

    expected_unreleased = (
        f"[Unreleased]: https://github.com/namidb/namidb/"
        f"compare/v{version}...HEAD"
    )
    if expected_unreleased not in changelog:
        errors.append(
            "CHANGELOG.md: [Unreleased] comparison must start at "
            f"v{version}"
        )
    release_link = re.compile(
        rf"^\[{re.escape(version)}\]:\s+\S*v{re.escape(version)}(?:\s|$)",
        re.MULTILINE,
    )
    if not release_link.search(changelog):
        errors.append(
            f"CHANGELOG.md: no comparison/release link exists for {version}"
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

    validate_workflow_action_pins(errors)
    validate_docker_base_pins(errors, rust_version)
    validate_documented_image_major(errors, version)

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
