"""Package the complete AdLift project into a clean, uploadable ZIP archive."""

import os
import zipfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_ZIP_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
OUTPUT_ZIP_PATH = os.path.join(OUTPUT_ZIP_DIR, "AdLift_Project_Complete.zip")

# Included directories
INCLUDE_DIRS = [
    "src",
    "scripts",
    "tests",
    "reports",
    "models",
    "docs",
    "app",
    os.path.join("data", "demo"),
]

# Included root files
INCLUDE_FILES = [
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    ".gitignore",
]

# Excluded patterns
EXCLUDE_DIRS = {
    "__pycache__",
    ".venv",
    "venv",
    ".git",
    ".pytest_cache",
    ".devcontainer",
    ".streamlit",
    "SecAgent-Mesh",
    "outputs",
}


def should_exclude(rel_path: str) -> bool:
    parts = rel_path.split(os.sep)
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.endswith(".pyc") or part.endswith(".pyo"):
            return True
    return False


def make_archive() -> str:
    print(f"Creating project ZIP archive from: {PROJECT_ROOT}")
    print(f"Destination: {OUTPUT_ZIP_PATH}")

    if os.path.exists(OUTPUT_ZIP_PATH):
        os.remove(OUTPUT_ZIP_PATH)

    with zipfile.ZipFile(OUTPUT_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add root files
        for fname in INCLUDE_FILES:
            fpath = os.path.join(PROJECT_ROOT, fname)
            if os.path.exists(fpath):
                zf.write(fpath, arcname=os.path.join("AdLift", fname))
                print(f"Added file: {fname}")

        # Add included directories
        for dname in INCLUDE_DIRS:
            dir_full = os.path.join(PROJECT_ROOT, dname)
            if not os.path.exists(dir_full):
                continue
            for root, dirs, files in os.walk(dir_full):
                # Filter out unwanted dirs in-place
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for file in files:
                    file_full = os.path.join(root, file)
                    rel_to_proj = os.path.relpath(file_full, PROJECT_ROOT)
                    if should_exclude(rel_to_proj):
                        continue
                    arcname = os.path.join("AdLift", rel_to_proj)
                    zf.write(file_full, arcname=arcname)

    size_mb = os.path.getsize(OUTPUT_ZIP_PATH) / (1024 * 1024)
    print(f"\nSuccessfully created archive: {OUTPUT_ZIP_PATH}")
    print(f"Archive Size: {size_mb:.2f} MB")
    return OUTPUT_ZIP_PATH


if __name__ == "__main__":
    make_archive()
