#!/usr/bin/env python3
"""兼容入口：请改用 applications.apply_ranker"""

from applications.apply_ranker import apply_ltr_ranking, main  # noqa: F401


if __name__ == "__main__":
    main()
