#!/usr/bin/env python3
"""兼容入口：请改用 applications.run_ranking_pipeline"""

import sys

from applications.run_ranking_pipeline import main  # noqa: F401


if __name__ == "__main__":
    sys.exit(main())
