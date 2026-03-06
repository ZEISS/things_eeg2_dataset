"""Verifies dependencies for the THINGS-EEG2 dataset CLI application."""
from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path
from packaging.specifiers import SpecifierSet
from packaging.version import Version

CRITICAL_DEPENDENCIES = {
    "transformers": ">=4.51.0,<4.52.0",
    "diffusers": ">=0.32.0,<0.37.0",
    "torch": ">=2.0.0",
    "torchvision": ">=0.24.1",
    "mne": ">=1.11.0",
    "pandas": ">=2.3.3",
}


def find_project_root() -> Path | None:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def get_installed_version(package: str) -> str | None:
    """Get installed version of a package."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_version(package: str, required: str) -> tuple[bool, str]:
    """Check if installed package version meets requirements."""
    installed = get_installed_version(package)
    if not installed:
        return False, f"{package} not installed (required: {required})"
    
    try:
        if Version(installed) in SpecifierSet(required):
            return True, f"{package} compatible (v{installed})"
        return False, f"{package} v{installed} doesn't satisfy {required}"
    except Exception as e:
        return False, f"Error checking {package}: {e}"


def sync_with_uv(project_root: Path) -> bool:
    """Sync dependencies using uv."""
    try:
        print("Syncing dependencies with uv...\n")
        result = subprocess.run(["uv", "sync"], cwd=project_root, check=False)
        
        if result.returncode == 0:
            print("Successfully synced dependencies! \n")
            return True
        
        print(f"uv sync failed (exit code {result.returncode}) \n")
        return False
    except FileNotFoundError:
        print(f"'uv' command not found \n")
        return False
    except Exception as e:
        print(f"Error running uv sync: {e} \n")
        return False


def auto_verify_and_install_dependencies(silent: bool = False) -> bool:
    """
    Verify dependencies and auto-install if needed using uv.
    
    Args:
        silent: Suppress output unless there's an error
        
    Returns:
        True if all dependencies satisfied, False otherwise
    """
    # Check which packages need fixing
    incompatible = [
        (pkg, ver) for pkg, ver in CRITICAL_DEPENDENCIES.items()
        if not check_version(pkg, ver)[0]
    ]
    
    if not incompatible:
        return True
    
    if not silent:
        print("Dependency issues detected. Attempting to fix...\n")
    
    # Verify uv is available
    if not shutil.which("uv"):
        print("Error: 'uv' is required but not installed. Install uv first.\n")
        return False
    
    # Find project root
    project_root = find_project_root()
    if not project_root:
        print(f"Error: Could not find project root (pyproject.toml).\n")
        return False
    
    # Run uv sync
    if not silent:
        print("Using uv to sync dependencies\n")
    
    if not sync_with_uv(project_root):
        print("uv sync failed.\n")
        print("Try running manually: \n")
        print(f"cd {project_root} \n")
        print("uv sync \n")
        return False
    
    # Verify sync worked
    remaining = [
        (pkg, msg) for pkg, ver in incompatible
        if not (is_ok := check_version(pkg, ver))[0]
        for msg in [is_ok[1]]
    ]
    
    if not remaining:
        if not silent:
            print("All dependencies are now compatible!\n")
        return True
    
    # uv synced but versions still differ (might be OK per lockfile)
    if not silent:
        print("\n uv sync succeeded, but some versions differ:")
        for pkg, msg in remaining:
            print(f"  • {msg}")
    
    return True


def verify_dependencies() -> None:
    """Verify all critical dependencies and print results."""
    print("Dependency Verification Report:\n")
    
    results = [check_version(pkg, ver) for pkg, ver in CRITICAL_DEPENDENCIES.items()]
    all_passed = all(ok for ok, _ in results)
    
    for (is_ok, msg) in results:
        status = "✓" if is_ok else "✗"
        print(f"{status} {msg}")
    
    if all_passed:
        print("All dependencies verified successfully!\n")
        return
    
    print("Some dependencies are missing or incompatible.\n")
    
    if not shutil.which("uv"):
        print("'uv' is not installed. Install uv first.\n")
        return
    
    if project_root := find_project_root():
        print("To fix dependencies, run:\n")
        print(f"cd {project_root}\n")
        print("uv sync\n")
    else:
        print("Could not find project root.\n")

if __name__ == "__main__":
    verify_dependencies()