import sys
from pathlib import Path

# Ensure src is on sys.path for package imports
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / 'src'
if str(src_path) not in sys.path:
	sys.path.insert(0, str(src_path))
