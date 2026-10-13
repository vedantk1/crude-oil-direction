"""Create a fresh pinned environment; never modify an existing environment."""

import argparse
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import venv


SCIPY_MACOS12_WHEEL = "scipy-1.15.3-cp313-cp313-macosx_12_0_arm64.whl"
SCIPY_MACOS12_SHA256 = "7e11270a000969409d37ed399585ee530b9ef6aa99d50c019de4cb01e8e54e62"


def checked_environment(value):
    path = Path(value)
    if (path.is_absolute() or ".." in path.parts or path == Path(".")
            or not path.resolve().is_relative_to(Path.cwd().resolve())):
        raise ValueError("Environment must use a workspace-relative path without traversal")
    if path.exists():
        raise ValueError("Choose a new environment directory; existing environments are preserved")
    return path


def bootstrap(environment=Path(".venv")):
    if sys.version_info[:2] != (3, 13):
        raise ValueError("Use Python 3.13 for the pinned, tested environment")
    environment = checked_environment(environment)
    requirements = Path("requirements.txt")
    if not requirements.is_file():
        raise ValueError("Run bootstrap from the repository root")
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    pip = [str(python), "-m", "pip"]
    # The macOS 14 arm64 SciPy wheel can fail to load on some hosts. Select
    # the official, same-version macOS 12 build before installation instead.
    compatible_scipy = sys.platform == "darwin" and platform.machine() == "arm64"
    with tempfile.TemporaryDirectory(prefix="oil-bootstrap-") as temporary:
        root = Path(temporary)
        selected = root / "requirements.txt"
        selected.write_text("\n".join(line for line in requirements.read_text().splitlines()
                                     if not compatible_scipy or not line.startswith("scipy==")) + "\n")
        if compatible_scipy:
            subprocess.run([*pip, "download", "--only-binary=:all:", "--no-deps",
                            "--platform", "macosx_12_0_arm64", "--python-version", "313",
                            "--implementation", "cp", "--abi", "cp313", "--dest", str(root),
                            "--index-url", "https://pypi.org/simple", "scipy==1.15.3"], check=True)
            wheel = root / SCIPY_MACOS12_WHEEL
            if not wheel.is_file() or hashlib.sha256(wheel.read_bytes()).hexdigest() != SCIPY_MACOS12_SHA256:
                raise ValueError("Official SciPy wheel checksum differs from the documented build")
            # Install SciPy before sklearn so its resolver retains this build.
            subprocess.run([*pip, "install", "--index-url", "https://pypi.org/simple",
                            "numpy==2.2.6", str(wheel)], check=True)
        subprocess.run([*pip, "install", "--index-url", "https://pypi.org/simple",
                        "-r", str(selected)], check=True)
    subprocess.run([str(python), "-c", "import numpy, scipy, sklearn; from scipy import linalg; "
                    "print('Pinned numeric runtime imported successfully')"], check=True)
    print("Environment ready:", str(environment))
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=Path(".venv"))
    args = parser.parse_args()
    bootstrap(args.environment)


if __name__ == "__main__":
    main()
