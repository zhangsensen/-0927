#!/usr/bin/env python3
"""
Daily schedule checker.
Reads operations/schedule.yaml and a state file operations/schedule_state.json.
Outputs whether rebalance (every N days) or strategy review (every M days) is due.

State file fields:
  last_rebalance: YYYY-MM-DD
  last_strategy_review: YYYY-MM-DD

CLI options:
  --mark-rebalance    Update last_rebalance to today
  --mark-review       Update last_strategy_review to today
  --state             Custom state file path
  --schedule          Custom schedule yaml path
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import date, datetime
import sys

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

DEFAULT_SCHEDULE = Path('operations/schedule.yaml')
DEFAULT_STATE = Path('operations/schedule_state.json')


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError('PyYAML not installed')
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def days_since(dstr: str | None) -> int | None:
    if not dstr:
        return None
    try:
        dt = datetime.strptime(dstr, '%Y-%m-%d').date()
        return (date.today() - dt).days
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schedule', default=str(DEFAULT_SCHEDULE))
    ap.add_argument('--state', default=str(DEFAULT_STATE))
    ap.add_argument('--mark-rebalance', action='store_true')
    ap.add_argument('--mark-review', action='store_true')
    args = ap.parse_args()

    schedule = load_yaml(Path(args.schedule))
    state = load_state(Path(args.state))

    rebalance_freq = schedule.get('rebalance_every_days') or schedule.get('rebalance_every_days'.replace('-', '_'))
    review_freq = schedule.get('strategy_review_every_days')

    last_reb = state.get('last_rebalance')
    last_rev = state.get('last_strategy_review')
    ds_reb = days_since(last_reb)
    ds_rev = days_since(last_rev)

    today_str = date.today().strftime('%Y-%m-%d')

    due_rebalance = False
    due_review = False

    if rebalance_freq is not None:
        if ds_reb is None or ds_reb >= rebalance_freq:
            due_rebalance = True
    if review_freq is not None:
        if ds_rev is None or ds_rev >= review_freq:
            due_review = True

    if args.mark_rebalance:
        state['last_rebalance'] = today_str
        ds_reb = 0
        due_rebalance = False
    if args.mark_review:
        state['last_strategy_review'] = today_str
        ds_rev = 0
        due_review = False

    # Prepare output
    report = {
        'date': today_str,
        'rebalance_freq_days': rebalance_freq,
        'strategy_review_freq_days': review_freq,
        'days_since_rebalance': ds_reb,
        'days_since_strategy_review': ds_rev,
        'due_rebalance': due_rebalance,
        'due_strategy_review': due_review,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # Persist state when marking actions
    if args.mark_rebalance or args.mark_review or not Path(args.state).exists():
        save_state(Path(args.state), state)

    # exit code summarizing urgency (2 if both due, 1 if any due, 0 if none)
    code = 0
    if due_rebalance and due_review:
        code = 2
    elif due_rebalance or due_review:
        code = 1
    sys.exit(code)


if __name__ == '__main__':
    main()
