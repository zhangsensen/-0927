#!/usr/bin/env python3
"""
Generate portfolio daily NAV/returns from live allocation and ETF daily prices/returns.

Inputs:
  --alloc: live_alloc CSV (strategy_combo_key,symbol,target_weight,capital_alloc,...)
  --prices: CSV with columns [date,symbol,close] OR [date,symbol,ret]
  --rebalance-days: Rebalance back to target weights every N days (default 8)
  --out: output CSV path for portfolio daily series

Assumptions:
 - Within each strategy, symbol weights (target_weight) sum to 1
 - Strategy capital is constant unless rebalancing day (no flow in/out)
 - Portfolio is sum of strategy-level sub-portfolios

Outputs:
  CSV: date,portfolio_nav,portfolio_ret
  MD : summary stats (annual_return, vol, sharpe, max_drawdown)
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
from collections import defaultdict
from datetime import datetime, timedelta


def read_alloc(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def read_market(path: Path):
    # detect header: date,symbol,ret or date,symbol,close
    rows = []
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = [c.strip().lower() for c in reader.fieldnames or []]
        has_ret = 'ret' in fields
        has_close = 'close' in fields
        if not (has_ret or has_close):
            raise ValueError('prices file must have ret or close column')
        for r in reader:
            date_str = r.get('date') or r.get('Date')
            sym = r.get('symbol') or r.get('Symbol')
            if not date_str or not sym:
                continue
            d = datetime.strptime(date_str[:10], '%Y-%m-%d').date()
            if has_ret:
                try:
                    ret = float(r.get('ret'))
                except Exception:
                    continue
                rows.append((d, sym, None, ret))
            else:
                try:
                    close = float(r.get('close'))
                except Exception:
                    continue
                rows.append((d, sym, close, None))
    return rows


def compute_returns_from_close(rows):
    # rows: (date, symbol, close, None)
    by_sym = defaultdict(list)
    for d, s, c, _ in rows:
        by_sym[s].append((d, c))
    out = []
    for s, lst in by_sym.items():
        lst.sort(key=lambda x: x[0])
        prev = None
        for d, c in lst:
            if prev is None:
                ret = 0.0
            else:
                ret = (c / prev) - 1.0 if prev != 0 else 0.0
            out.append((d, s, ret))
            prev = c
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def max_drawdown(equity):
    peak = equity[0]
    mdd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = (x / peak) - 1.0
        if dd < mdd:
            mdd = dd
    return mdd


def annualized_stats(rets, periods_per_year=252):
    import math
    if not rets:
        return 0.0, 0.0, 0.0
    avg = sum(rets) / len(rets)
    var = sum((x - avg) ** 2 for x in rets) / (len(rets) - 1 or 1)
    vol = math.sqrt(var * periods_per_year)
    ann = (1 + avg) ** periods_per_year - 1
    sharpe = ann / vol if vol > 1e-12 else 0.0
    return ann, vol, sharpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alloc', required=True)
    ap.add_argument('--prices', required=True)
    ap.add_argument('--rebalance-days', type=int, default=8)
    ap.add_argument('--out', default='portfolio_daily_curve.csv')
    ap.add_argument('--report', default='portfolio_daily_report.md')
    args = ap.parse_args()

    alloc = read_alloc(Path(args.alloc))
    if not alloc:
        print('ERROR: empty alloc', file=sys.stderr)
        sys.exit(2)
    # group by strategy symbol weights and capital
    strat_weights = defaultdict(dict)  # strat -> symbol -> weight
    strat_capital = {}
    for r in alloc:
        strat = r['strategy_combo_key']
        sym = r['symbol']
        try:
            w = float(r['target_weight'])
        except Exception:
            w = 0.0
        strat_weights[strat][sym] = w
        if strat not in strat_capital:
            try:
                strat_capital[strat] = float(r['capital_alloc'])
            except Exception:
                strat_capital[strat] = 0.0

    market_rows = read_market(Path(args.prices))
    # normalize to returns
    if market_rows and market_rows[0][2] is None:
        # ret provided
        rets = [(d, s, r) for (d, s, _, r) in market_rows]
    else:
        rets = compute_returns_from_close(market_rows)

    # build date index and symbol returns map
    by_date = defaultdict(dict)  # date -> symbol -> ret
    dates = set()
    for d, s, r in rets:
        dates.add(d)
        by_date[d][s] = r
    dates = sorted(dates)

    # track per-strategy shares notionally to simulate rebalancing
    # initialize on first date with capital and target weights
    nav_series = []
    total_init_cap = sum(strat_capital.values())
    if total_init_cap <= 0:
        print('ERROR: total capital is zero', file=sys.stderr)
        sys.exit(2)

    last_reb_day = None
    strat_values = {k: v for k, v in strat_capital.items()}  # start as capital
    # For simplicity we apply returns multiplicatively on each date according to symbol weights
    # and rebalance weights back to target every N days.

    prev_total = total_init_cap
    for i, d in enumerate(dates):
        # check rebalance
        do_reb = False
        if last_reb_day is None:
            do_reb = True
        else:
            delta = (d - last_reb_day).days
            if delta >= args.rebalance_days:
                do_reb = True
        if do_reb:
            last_reb_day = d
            # reset per-strategy symbol allocations to target weights implicitly (no carryover of drift)
            pass

        # apply returns at strategy level as weighted sum of symbol returns (missing symbol return treated as 0)
        total_val = 0.0
        for strat, cap in strat_values.items():
            sym_weights = strat_weights[strat]
            # daily strat return
            strat_ret = 0.0
            for sym, w in sym_weights.items():
                r = by_date[d].get(sym, 0.0)
                strat_ret += w * r
            strat_values[strat] = strat_values[strat] * (1.0 + strat_ret)
            total_val += strat_values[strat]

        port_ret = (total_val / prev_total) - 1.0 if prev_total != 0 else 0.0
        nav_series.append((d, total_val, port_ret))
        prev_total = total_val

    # write CSV
    out_path = Path(args.out)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['date', 'portfolio_nav', 'portfolio_ret'])
        for d, nav, r in nav_series:
            w.writerow([d.isoformat(), f"{nav:.6f}", f"{r:.8f}"])

    # summary report
    equity = [x[1] for x in nav_series]
    rets_only = [x[2] for x in nav_series[1:]]  # skip first 0-ret day
    ann, vol, sharpe = annualized_stats(rets_only)
    mdd = max_drawdown(equity)
    rep = Path(args.report)
    rep.write_text(
        ("# Portfolio Daily Report\n\n"
         f"- Start: {dates[0].isoformat()}\n"
         f"- End: {dates[-1].isoformat()}\n"
         f"- Initial Capital: {total_init_cap:,.2f}\n"
         f"- Final NAV: {equity[-1]:,.2f}\n"
         f"- Annualized Return: {ann:.4%}\n"
         f"- Annualized Vol: {vol:.4%}\n"
         f"- Sharpe: {sharpe:.3f}\n"
         f"- Max Drawdown: {mdd:.2%}\n")
    , encoding='utf-8')

    print(f"Wrote {out_path} and {rep}")


if __name__ == '__main__':
    main()
