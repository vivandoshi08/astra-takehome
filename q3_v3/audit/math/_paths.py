"""Where the math audit scripts write scratch files (npz, png, cache sandboxes).
Never inside the repo: $Q3_AUDIT_OUT, else <tmp>/q3_audit/math."""
import os
import tempfile
from pathlib import Path

OUT = Path(os.environ.get("Q3_AUDIT_OUT",
                          Path(tempfile.gettempdir()) / "q3_audit")) / "math"
OUT.mkdir(parents=True, exist_ok=True)
