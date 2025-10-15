from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

CN_MORNING_START = time(9, 30)
CN_MORNING_END = time(11, 30)  # right-exclusive in slicing
CN_AFTERNOON_START = time(13, 0)
CN_AFTERNOON_END = time(15, 0)  # right-exclusive in slicing


def _load_holidays_file(path: Path) -> Set[date]:
    if not path.exists():
        return set()
    holidays: Set[date] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # supports YYYY-MM-DD or YYYYMMDD
        if len(s) == 8 and s.isdigit():
            d = date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
        else:
            d = date.fromisoformat(s)
        holidays.add(d)
    return holidays


class CNCalendar:
    """Minimal A-share trading calendar with lunch break sessions.

    - Weekends are non-trading by default.
    - Optional holiday file: config/cn_holidays.txt (one date per line).
    - Provides morning and afternoon sessions for each trading day.
    """

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = project_root or Path.cwd()
        holidays_path = self.project_root / "config" / "cn_holidays.txt"
        self._holidays = _load_holidays_file(holidays_path)

    def is_trading_day(self, d: date) -> bool:
        if d.weekday() >= 5:
            return False
        if d in self._holidays:
            return False
        return True

    def trading_sessions(self, d: date) -> List[Tuple[datetime, datetime]]:
        """Return session start/end datetimes (right-exclusive ends)."""
        if not self.is_trading_day(d):
            return []
        return [
            (
                datetime.combine(d, CN_MORNING_START),
                datetime.combine(d, CN_MORNING_END),
            ),
            (
                datetime.combine(d, CN_AFTERNOON_START),
                datetime.combine(d, CN_AFTERNOON_END),
            ),
        ]

    def iter_trading_days(self, start: date, end: date) -> Iterable[date]:
        cur = start
        while cur <= end:
            if self.is_trading_day(cur):
                yield cur
            cur += timedelta(days=1)
