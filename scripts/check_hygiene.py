"""Read-only validation of configuration and merge-conflict markers."""

from pathlib import Path
import subprocess
import yaml


def main():
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
        )
        .decode()
        .split("\0")
    )
    failures = []
    for name in sorted(set(filter(None, paths))):
        path = Path(name)
        if not path.is_file() or path.suffix not in {
            ".py",
            ".md",
            ".yaml",
            ".yml",
            ".toml",
            ".txt",
        }:
            continue
        content = path.read_text(encoding="utf-8-sig")
        if any(
            line.startswith(("<<<<<<< ", ">>>>>>> ")) for line in content.splitlines()
        ):
            failures.append(f"{name}: unresolved merge conflict")
        if path.suffix in {".yaml", ".yml"}:
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as error:
                failures.append(f"{name}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("YAML and conflict-marker checks passed.")


if __name__ == "__main__":
    main()
