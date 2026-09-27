"""Collect post-observation actor states on fresh evaluation episodes."""
import sys
from .evaluate import main

if __name__ == "__main__":
    if "--collect" not in sys.argv:
        sys.argv.append("--collect")
    if "--policy" not in sys.argv:
        sys.argv.extend(["--policy", "recurrent"])
    main()
