import re
import toml

def read_version_from_init():
    with open("hwpwn/__init__.py") as f:
        content = f.read()
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
    return match.group(1)

def read_version_from_file(filename):
    with open(filename, "r") as f:
        content = f.read()
    match = re.search(r"<version_placeholder>", content)
    return match.group(1)


def check_version_consistency():
    version_init = read_version_from_init()
    version_toml = read_version_from_toml()

    if version_init != version_toml:
        print(f"Version mismatch: {version_init} (in __init__.py) != {version_toml} (in pyproject.toml)")
        return False

    return True

def read_version_from_toml():
    with open("pyproject.toml") as f:
        content = toml.load(f)
    return content["tool"]["poetry"]["version"]

if __name__ == "__main__":
    if not check_version_consistency():
        raise SystemExit("Version mismatch detected.")

