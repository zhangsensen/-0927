#!/usr/bin/env python3
"""
QMT Bridge 端到端全功能验证脚本
用于检查与 QMT Bridge 服务的连接以及各主要数据接口的可用性。
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Any

try:
    from qmt_bridge import QMTClient, QMTClientConfig
except ImportError:
    print("❌ 错误: 未安装 qmt-data-bridge")
    print("请运行: pip install qmt-data-bridge")
    sys.exit(1)

# 配置
HOST = "192.168.122.132"
PORT = 8001
TEST_SYMBOL = "510300.SH"  # 沪深300ETF
TEST_INDEX = "000300.SH"  # 沪深300指数


def print_header(title: str):
    print(f"\n{'='*20} {title} {'='*20}")


def print_result(name: str, success: bool, data: Any = None, error: str = None):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {name}")
    if error:
        print(f"   Error: {error}")
    if data is not None:
        # 格式化打印数据摘要
        if isinstance(data, (dict, list)):
            try:
                json_str = json.dumps(data, ensure_ascii=False, default=str)
                if len(json_str) > 200:
                    print(f"   Data: {json_str[:200]}... (len={len(str(data))})")
                else:
                    print(f"   Data: {json_str}")
            except:
                print(f"   Data: {str(data)[:200]}...")
        else:
            print(f"   Data: {str(data)[:200]}...")


async def main():
    print(f"正在连接 QMT Bridge ({HOST}:{PORT})...")

    config = QMTClientConfig(host=HOST, port=PORT)
    client = QMTClient(config)

    results = []

    # 1. 测试 K线数据 (get_kline)
    print_header("测试: K线数据 (get_kline)")
    try:
        res = await client.get_kline(code=TEST_SYMBOL, period="1d", count=5)
        # 检查返回结构，通常包含 'bars' 或直接是列表
        bars = res.get("bars", []) if isinstance(res, dict) else res
        success = len(bars) > 0
        print_result("get_kline", success, data=f"获取到 {len(bars)} 条K线")
        results.append(success)
    except Exception as e:
        print_result("get_kline", False, error=str(e))
        results.append(False)

    # 2. 测试 Tick 数据 (get_tick)
    print_header("测试: Tick 数据 (get_tick)")
    try:
        res = await client.get_tick(code=TEST_SYMBOL)
        # Tick 数据通常是一个字典或对象
        success = res is not None
        print_result("get_tick", success, data=res)
        results.append(success)
    except Exception as e:
        print_result("get_tick", False, error=str(e))
        results.append(False)

    # 3. 测试 资产信息 (get_assets)
    print_header("测试: 资产信息 (get_assets)")
    try:
        res = await client.get_assets()
        success = res is not None
        print_result("get_assets", success, data=res)
        results.append(success)
    except Exception as e:
        print_result("get_assets", False, error=str(e))
        results.append(False)

    # 4. 测试 持仓信息 (get_positions)
    print_header("测试: 持仓信息 (get_positions)")
    try:
        res = await client.get_positions()
        # 持仓可能为空列表，但也算成功
        success = isinstance(res, list)
        print_result(
            "get_positions",
            success,
            data=f"持仓数量: {len(res) if res else 0}",
            error=None if success else "返回类型错误",
        )
        if success and res:
            print(f"   首个持仓: {res[0]}")
        results.append(success)
    except Exception as e:
        print_result("get_positions", False, error=str(e))
        results.append(False)

    # 5. 测试 委托信息 (get_orders)
    print_header("测试: 委托信息 (get_orders)")
    try:
        res = await client.get_orders()
        # 返回可能是 list 或 dict
        success = isinstance(res, (list, dict))
        count = len(res) if res else 0
        if isinstance(res, dict):
            # 如果是 dict，可能是 {"orders": [...]} 或 account_id -> orders
            count = sum(len(v) for v in res.values()) if res else 0

        print_result(
            "get_orders", success, data=f"委托结构: {type(res).__name__}, 数量: {count}"
        )
        results.append(success)
    except Exception as e:
        print_result("get_orders", False, error=str(e))
        results.append(False)

    # 6. 测试 成交信息 (get_trades)
    print_header("测试: 成交信息 (get_trades)")
    try:
        res = await client.get_trades()
        success = isinstance(res, (list, dict))
        count = len(res) if res else 0
        if isinstance(res, dict):
            count = sum(len(v) for v in res.values()) if res else 0

        print_result(
            "get_trades", success, data=f"成交结构: {type(res).__name__}, 数量: {count}"
        )
        results.append(success)
    except Exception as e:
        print_result("get_trades", False, error=str(e))
        results.append(False)

    # 7. 测试 板块成分股 (get_sector_stocks)
    print_header("测试: 板块成分股 (get_sector_stocks)")
    try:
        # 参数名修正: sector_name
        # 尝试使用 '000300.SH' 或 '沪深300'
        # 注意：QMT 的板块名称可能比较特殊，如果 000300.SH 失败，可以尝试 '沪深300'
        sector = TEST_INDEX
        res = await client.get_sector_stocks(sector_name=sector)
        success = isinstance(res, (list, dict))
        print_result(f"get_sector_stocks({sector})", success, data=res)
        results.append(success)
    except Exception as e:
        print_result("get_sector_stocks", False, error=str(e))
        # 再次尝试中文名
        try:
            print("   尝试中文名 '沪深300'...")
            res = await client.get_sector_stocks(sector_name="沪深300")
            success = isinstance(res, (list, dict))
            print_result("get_sector_stocks(沪深300)", success, data=res)
            if success:
                results.append(True)
        except:
            results.append(False)

    # 8. 测试 交易日历 (get_trading_calendar)
    print_header("测试: 交易日历 (get_trading_calendar)")
    try:
        today = datetime.now().strftime("%Y%m%d")
        # 参数名修正: market
        res = await client.get_trading_calendar(
            market="SH", start_time=today, end_time=today
        )
        success = res is not None
        print_result("get_trading_calendar", success, data=res)
        results.append(success)
    except Exception as e:
        print_result("get_trading_calendar", False, error=str(e))
        results.append(False)

    # 总结
    print_header("测试总结")
    total = len(results)
    passed = sum(results)
    print(f"总计测试: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")

    if passed == total:
        print("\n🎉 所有核心接口测试通过！QMT Bridge 服务连接正常且功能完备。")
    else:
        print("\n⚠️ 部分接口测试失败，请检查错误日志。")


if __name__ == "__main__":
    asyncio.run(main())
