#!/usr/bin/env python3
"""Launch the React dashboard dev server, or build and open a static dashboard.

Usage:
    python ablations/viewer.py          # build + open
    python ablations/viewer.py --dev    # launch vite dev server
"""

import subprocess
import sys
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


def main():
    if not DASHBOARD_DIR.exists():
        print(f"Dashboard not found at {DASHBOARD_DIR}.", file=sys.stderr)
        sys.exit(1)

    if "--dev" in sys.argv:
        subprocess.run(["npm", "run", "dev"], cwd=DASHBOARD_DIR)
    else:
        print("Building dashboard...")
        result = subprocess.run(
            ["npm", "run", "build"], cwd=DASHBOARD_DIR, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            sys.exit(1)
        dist = DASHBOARD_DIR / "dist" / "index.html"
        print(f"Dashboard built at {dist}")
        webbrowser.open(dist.as_uri())


if __name__ == "__main__":
    main()
