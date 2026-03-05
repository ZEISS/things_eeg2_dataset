import pytest
import warnings

import sys
from pathlib import Path


# Ensure tests run against this repository's sources (src/) rather than an
# unrelated installed package from another workspace/venv.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

@pytest.fixture(autouse=True)
def filter_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='torch.jit._script')