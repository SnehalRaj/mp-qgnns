"""Put ``src`` on the path so ``pytest`` works without installing the package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
