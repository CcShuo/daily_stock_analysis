# -*- coding: utf-8 -*-
"""
Tests for fundamental adapter helpers.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.fundamental_adapter import (
    AkshareFundamentalAdapter,
    _TushareFundamentalClient,
    _build_dividend_payload,
    _extract_latest_row,
    _has_meaningful_payload,
    _parse_dividend_plan_to_per_share,
)


class TestFundamentalAdapter(unittest.TestCase):
    def test_parse_dividend_plan_to_per_share_supports_cn_patterns(self) -> None:
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("10派3元(含税)"), 0.3, places=6)
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("每10股派发2.5元"), 0.25, places=6)
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("每股派0.8元"), 0.8, places=6)
        self.assertIsNone(_parse_dividend_plan_to_per_share("仅送股，不现金分红"))

    def test_extract_latest_row_returns_none_when_code_mismatch(self) -> None:
        df = pd.DataFrame(
            {
                "股票代码": ["600000", "000001"],
                "值": [1, 2],
            }
        )
        row = _extract_latest_row(df, "600519")
        self.assertIsNone(row)

    def test_extract_latest_row_fallback_when_no_code_column(self) -> None:
        df = pd.DataFrame({"值": [1, 2]})
        row = _extract_latest_row(df, "600519")
        self.assertIsNotNone(row)
        self.assertEqual(row["值"], 1)

    def test_dragon_tiger_no_match_with_code_column_is_ok(self) -> None:
        adapter = AkshareFundamentalAdapter()
        df = pd.DataFrame(
            {
                "股票代码": ["600000"],
                "日期": ["2026-01-01"],
            }
        )
        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_lhb_stock_statistic_em", [])):
            result = adapter.get_dragon_tiger_flag("600519")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["is_on_list"])
        self.assertEqual(result["recent_count"], 0)

    def test_dragon_tiger_match_is_ok(self) -> None:
        adapter = AkshareFundamentalAdapter()
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "日期": [today],
            }
        )
        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_lhb_stock_statistic_em", [])):
            result = adapter.get_dragon_tiger_flag("600519")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["is_on_list"])
        self.assertGreaterEqual(result["recent_count"], 1)

    def test_fundamental_bundle_includes_financial_report_and_dividend_payload(self) -> None:
        adapter = AkshareFundamentalAdapter()
        now = datetime.now()
        within_ttm = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        future_day = (now + timedelta(days=10)).strftime("%Y-%m-%d")
        old_day = (now - timedelta(days=500)).strftime("%Y-%m-%d")
        fin_df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "报告期": [within_ttm],
                "营业总收入": [1000.0],
                "归母净利润": [300.0],
                "经营活动产生的现金流量净额": [500.0],
                "净资产收益率": [18.2],
                "营业收入同比": [12.0],
                "净利润同比": [9.5],
            }
        )
        forecast_df = pd.DataFrame({"股票代码": ["600519"], "预告": ["预增"]})
        quick_df = pd.DataFrame({"股票代码": ["600519"], "快报": ["快报摘要"]})
        dividend_df = pd.DataFrame(
            {
                "股票代码": ["600519", "600519", "600519", "600519"],
                "除息日": [within_ttm, within_ttm, future_day, old_day],
                "分配方案": ["10派3元(含税)", "10派3元(含税)", "10派5元", "10派1元"],
            }
        )

        with patch.object(
            adapter,
            "_call_df_candidates",
            side_effect=[
                (fin_df, "stock_financial_abstract", []),
                (forecast_df, "stock_yjyg_em", []),
                (quick_df, "stock_yjkb_em", []),
                (dividend_df, "stock_fhps_detail_em", []),
                (None, None, []),
                (None, None, []),
            ],
        ):
            result = adapter.get_fundamental_bundle("600519")

        financial_report = result["earnings"].get("financial_report", {})
        self.assertEqual(financial_report.get("report_date"), within_ttm)
        self.assertEqual(financial_report.get("revenue"), 1000.0)
        self.assertEqual(financial_report.get("net_profit_parent"), 300.0)
        self.assertEqual(financial_report.get("operating_cash_flow"), 500.0)
        self.assertEqual(financial_report.get("roe"), 18.2)

        dividend_payload = result["earnings"].get("dividend", {})
        events = dividend_payload.get("events", [])
        self.assertEqual(len(events), 2)  # duplicate + future day filtered
        self.assertEqual(dividend_payload.get("ttm_event_count"), 1)
        self.assertAlmostEqual(dividend_payload.get("ttm_cash_dividend_per_share"), 0.3, places=6)

    def test_build_dividend_payload_returns_empty_when_code_not_matched(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["000001"],
                "除息日": [now],
                "分配方案": ["10派3元(含税)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload, {})

    def test_build_dividend_payload_skips_after_tax_plan(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "除息日": [now],
                "分配方案": ["10派3元(税后)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload, {})

    def test_build_dividend_payload_ttm_window_boundary(self) -> None:
        now = datetime.now()
        day_365 = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        day_366 = (now - timedelta(days=366)).strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519", "600519"],
                "除息日": [day_365, day_366],
                "分配方案": ["10派3元(含税)", "10派5元(含税)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload.get("ttm_event_count"), 1)
        self.assertAlmostEqual(payload.get("ttm_cash_dividend_per_share"), 0.3, places=6)

    def test_has_meaningful_payload_ignores_all_none_values(self) -> None:
        self.assertFalse(_has_meaningful_payload({"a": None, "b": {"c": None}}))
        self.assertTrue(_has_meaningful_payload({"a": None, "b": {"c": 1}}))

    def test_tushare_fundamental_bundle_prefers_structured_financials(self) -> None:
        adapter = AkshareFundamentalAdapter()
        response_payloads = [
            {
                "code": 0,
                "data": {
                    "fields": [
                        "ts_code",
                        "ann_date",
                        "end_date",
                        "roe",
                        "grossprofit_margin",
                        "tr_yoy",
                        "netprofit_yoy",
                    ],
                    "items": [["600519.SH", "20260420", "20260331", 18.2, 91.5, 12.0, 9.5]],
                },
            },
            {
                "code": 0,
                "data": {
                    "fields": ["ts_code", "ann_date", "end_date", "total_revenue", "revenue", "n_income_attr_p"],
                    "items": [["600519.SH", "20260420", "20260331", 1000.0, 990.0, 300.0]],
                },
            },
            {
                "code": 0,
                "data": {
                    "fields": ["ts_code", "ann_date", "end_date", "n_cashflow_act"],
                    "items": [["600519.SH", "20260420", "20260331", 500.0]],
                },
            },
            {
                "code": 0,
                "data": {
                    "fields": ["ts_code", "ann_date", "end_date", "cash_div_tax", "record_date", "ex_date"],
                    "items": [["600519.SH", "20260420", "20251231", 3.0, "20260520", "20260521"]],
                },
            },
        ]
        responses = [
            MagicMock(status_code=200, text=__import__("json").dumps(payload))
            for payload in response_payloads
        ]
        config = SimpleNamespace(tushare_token="demo-token", tushare_api_url="http://relay.example.com/")

        with patch("src.config.get_config", return_value=config), \
                patch("data_provider.fundamental_adapter.requests.post", side_effect=responses) as post_mock, \
                patch.object(
                    adapter,
                    "_call_df_candidates",
                    return_value=(None, None, []),
                ):
            result = adapter.get_fundamental_bundle("600519")

        financial_report = result["earnings"].get("financial_report", {})
        self.assertEqual(financial_report.get("report_date"), "2026-03-31")
        self.assertEqual(financial_report.get("revenue"), 1000.0)
        self.assertEqual(financial_report.get("net_profit_parent"), 300.0)
        self.assertEqual(financial_report.get("operating_cash_flow"), 500.0)
        self.assertEqual(financial_report.get("roe"), 18.2)
        self.assertEqual(result["growth"].get("revenue_yoy"), 12.0)
        self.assertEqual(result["growth"].get("net_profit_yoy"), 9.5)
        self.assertIn("financial_report:tushare_income_cashflow_fina_indicator", result["source_chain"])
        self.assertTrue(any(call.kwargs["json"]["api_name"] == "income" for call in post_mock.call_args_list))

    def test_tushare_client_posts_to_configured_endpoint(self) -> None:
        client = _TushareFundamentalClient("demo-token", "http://relay.example.com/", timeout=12)
        response = MagicMock(
            status_code=200,
            text='{"code":0,"data":{"fields":["ts_code"],"items":[["600519.SH"]]}}',
        )
        with patch("data_provider.fundamental_adapter.requests.post", return_value=response) as post_mock:
            df = client.query("income", params={"ts_code": "600519.SH"}, fields="ts_code")
        self.assertEqual(df.iloc[0]["ts_code"], "600519.SH")
        post_mock.assert_called_once_with(
            "http://relay.example.com/",
            json={"api_name": "income", "token": "demo-token", "params": {"ts_code": "600519.SH"}, "fields": "ts_code"},
            timeout=12,
        )


if __name__ == "__main__":
    unittest.main()
