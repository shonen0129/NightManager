"""TachibanaBrokerClient — BrokerClient adapter for Tachibana Securities API.

Adapts TachibanaClient to conform to the BrokerClient interface.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any

from leadlag.broker.base import BrokerClient, BrokerConfig, Position, WalletInfo
from leadlag.broker.tachibana.api import TachibanaApiError, TachibanaClient
from leadlag.config import TachibanaApiConfig
from leadlag.core.types import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)

logger = logging.getLogger(__name__)


class TachibanaBrokerClient(BrokerClient):
    """BrokerClient implementation backed by Tachibana Securities API.

    Translates between the broker-neutral interface and Tachibana-specific types.
    """

    def __init__(self, config: BrokerConfig) -> None:
        self._broker_config = config
        self._api_config = self._build_api_config(config)
        self._client = TachibanaClient(self._api_config)

    def _build_api_config(self, config: BrokerConfig) -> TachibanaApiConfig:
        """Translate neutral BrokerConfig to TachibanaApiConfig."""
        # Check config parameters with fallback to environment variables
        auth_id = (
            config.api_token
            or config.extra.get("auth_id")
            or os.environ.get("TACHIBANA_AUTH_ID", "")
        )
        private_key_path = (
            config.extra.get("private_key_path")
            or os.environ.get("TACHIBANA_PRIVATE_KEY_PATH", "")
        )
        
        # Security check: verify private key file has restrictive permissions
        if private_key_path and os.path.exists(private_key_path):
            import stat
            file_stat = os.stat(private_key_path)
            file_mode = stat.filemode(file_stat.st_mode)
            # Check if file is readable by group or others (should be 600 or 400)
            if file_stat.st_mode & (stat.S_IRGRP | stat.S_IROTH):
                logger.warning(
                    "SECURITY WARNING: Private key file %s has overly permissive permissions (%s). "
                    "Recommended: chmod 600 (owner read/write only)",
                    private_key_path, file_mode
                )
        second_password = (
            config.api_password
            or config.extra.get("second_password")
            or os.environ.get("TACHIBANA_SECOND_PASSWORD", "")
        )

        return TachibanaApiConfig(
            api_url=config.api_url or "https://kabuka.e-shiten.jp/e_api_v4r9",
            auth_id=auth_id,
            private_key_path=private_key_path,
            second_password=second_password,
            request_timeout=config.request_timeout,
            margin_trade_type=config.margin_trade_type,
            account_type=config.account_type,
        )

    def close(self) -> None:
        self._client.close()

    def get_order_detail(self, order_number: str, eigyou_day: str) -> dict[str, Any]:
        """Fetch order detail with fill information (約定価格、約定株数).

        Args:
            order_number: Order ID from submit_order response
            eigyou_day: Business day string (YYYYMMDD)

        Returns:
            Dict with sYakuzyouPrice, sYakuzyouSuryou, sYakuzyouDate,
            aYakuzyouSikkouList (partial fills), aKessaiOrderTategyokuList (settlement details).
        """
        return self._client.get_order_detail(order_number, eigyou_day)

    def health_check(self) -> bool:
        """Check connection by attempting to login and retrieve price for a test ticker."""
        try:
            if not self._client.logged_in:
                self._client.login()
            # Perform a test query using TOPIX ticker or sector ETF (e.g. "1617")
            res = self._client.get_price(["1617"])
            return len(res) > 0
        except Exception as e:
            logger.error("Tachibana health check failed: %s", e)
            return False

    def get_wallet(self) -> WalletInfo:
        """Retrieve cash and margin balance information."""
        res = self._client.get_wallet()
        cash = float(res.get("sGenbutuKabuKaituke", 0.0))
        margin = float(res.get("sSinyouSinkidate", 0.0))

        # Fetch margin detail for 受入保証金 (deposited margin = equity base)
        ukeire_hosyoukin = None
        hosyoukin_yoryoku = None
        hosyoukin_ritu = None
        try:
            detail = self._client.get_margin_detail(hituke_index=0)
            ukeire_hosyoukin = float(detail.get("sUkeireHosyoukin", 0.0))
            hosyoukin_yoryoku = float(detail.get("sHosyoukinYoryoku", 0.0))
            hosyoukin_ritu = float(detail.get("sHosyoukinRitu", 0.0))
        except Exception as e:
            logger.warning("Failed to fetch margin detail: %s", e)

        return WalletInfo(
            cash_available=cash,
            margin_available=margin,
            extra={
                "ukeire_hosyoukin": ukeire_hosyoukin,
                "hosyoukin_yoryoku": hosyoukin_yoryoku,
                "hosyoukin_ritu": hosyoukin_ritu,
                "sHosyouKinritu": res.get("sHosyouKinritu"),
                "sOisyouHasseiFlg": res.get("sOisyouHasseiFlg"),
                "sTatekaekinHasseiFlg": res.get("sTatekaekinHasseiFlg"),
            },
        )

    def fetch_open_prices(
        self,
        tickers: list[str],
        *,
        allow_missing: bool = False,
    ) -> dict[str, float]:
        """Fetch today's open prices for JP tickers (e.g. '1617.T')."""
        opens: dict[str, float] = {}
        missing_tickers: list[str] = []

        # Map to Tachibana symbols (without suffix)
        mapping = {tk.replace(".T", ""): tk for tk in tickers}
        clean_codes = list(mapping.keys())

        # Tachibana accepts up to 120 codes per query
        chunk_size = 100
        for i in range(0, len(clean_codes), chunk_size):
            chunk = clean_codes[i : i + chunk_size]
            try:
                res = self._client.get_price(chunk)
                for item in res:
                    code = item.get("sIssueCode")
                    p_dop_str = item.get("pDOP") # Opening Price
                    original_ticker = mapping.get(code)
                    if not original_ticker:
                        continue

                    try:
                        p_dop = float(p_dop_str) if p_dop_str and p_dop_str != "0.0000" else 0.0
                    except ValueError:
                        p_dop = 0.0

                    if p_dop > 0.0:
                        opens[original_ticker] = p_dop
                    else:
                        missing_tickers.append(original_ticker)
            except Exception as e:
                logger.error("Failed to fetch price chunk %s: %s", chunk, e)
                for code in chunk:
                    missing_tickers.append(mapping[code])

        # If any ticker was missed or had <= 0 open price
        # Make a second pass to record missing ones
        for tk in tickers:
            if tk not in opens:
                if tk not in missing_tickers:
                    missing_tickers.append(tk)

        if missing_tickers:
            msg = f"Failed to fetch open prices for {len(missing_tickers)} ticker(s): {', '.join(missing_tickers)}"
            if allow_missing:
                logger.warning(msg)
            else:
                logger.error(msg)
                raise ValueError(msg)

        return opens

    def fetch_current_prices(
        self,
        tickers: list[str],
        *,
        allow_missing: bool = False,
    ) -> dict[str, float]:
        """Fetch current real-time prices for JP tickers (e.g. '1617.T')."""
        prices: dict[str, float] = {}
        missing_tickers: list[str] = []

        mapping = {tk.replace(".T", ""): tk for tk in tickers}
        clean_codes = list(mapping.keys())

        chunk_size = 100
        for i in range(0, len(clean_codes), chunk_size):
            chunk = clean_codes[i : i + chunk_size]
            try:
                res = self._client.get_price(chunk)
                for item in res:
                    code = item.get("sIssueCode")
                    p_dpp_str = item.get("pDPP")  # Current Price
                    original_ticker = mapping.get(code)
                    if not original_ticker:
                        continue

                    try:
                        p_dpp = float(p_dpp_str) if p_dpp_str and p_dpp_str != "0.0000" else 0.0
                    except ValueError:
                        p_dpp = 0.0

                    if p_dpp > 0.0:
                        prices[original_ticker] = p_dpp
                    else:
                        missing_tickers.append(original_ticker)
            except Exception as e:
                logger.error("Failed to fetch price chunk %s: %s", chunk, e)
                for code in chunk:
                    missing_tickers.append(mapping[code])

        for tk in tickers:
            if tk not in prices and tk not in missing_tickers:
                missing_tickers.append(tk)

        if missing_tickers:
            msg = f"Failed to fetch current prices for {len(missing_tickers)} ticker(s): {', '.join(missing_tickers)}"
            if allow_missing:
                logger.warning(msg)
            else:
                logger.error(msg)
                raise ValueError(msg)

        return prices

    def fetch_us_etf_returns(
        self,
        us_tickers: list[str],
    ) -> dict[str, float]:
        """Fetch US ETF returns.

        Tachibana API is JP only; uses yfinance dynamically as a fallback.
        """
        import concurrent.futures

        logger.info("[TachibanaBroker] Fetching US ETF returns via yfinance fallback...")
        import yfinance as yf

        _YF_TIMEOUT = 30

        returns: dict[str, float] = {}
        failed: list[str] = []

        import threading

        for ticker in us_tickers:
            try:
                t_obj = yf.Ticker(ticker)
                result_box: dict = {}

                def _worker(_tobj=t_obj):
                    try:
                        result_box["value"] = _tobj.history(period="5d")
                    except Exception as e:
                        result_box["error"] = e

                th = threading.Thread(target=_worker, daemon=True)
                th.start()
                th.join(timeout=_YF_TIMEOUT)

                if th.is_alive():
                    logger.error("yfinance history() timed out for %s after %ds", ticker, _YF_TIMEOUT)
                    failed.append(ticker)
                    continue
                if "error" in result_box:
                    raise result_box["error"]
                hist = result_box["value"]

                if len(hist) >= 2:
                    last_close = hist["Close"].iloc[-1]
                    prev_close = hist["Close"].iloc[-2]
                    returns[ticker] = float(last_close / prev_close - 1.0)
                else:
                    failed.append(ticker)
            except Exception as e:
                logger.error("yfinance US return fetch failed for %s: %s", ticker, e)
                failed.append(ticker)

        if failed:
            logger.warning(
                "Failed to fetch returns for %d US ETF(s): %s",
                len(failed),
                ", ".join(failed),
            )
        logger.info("Successfully fetched returns for %d/%d US ETFs", len(returns), len(us_tickers))
        return returns

    def get_positions(self, **filters: Any) -> list[Position]:
        """Fetch current open credit positions."""
        raw = self._client.get_positions(ticker=filters.get("symbol"))
        positions: list[Position] = []

        for pos in raw:
            issue_code = str(pos.get("sOrderIssueCode", ""))
            ticker = f"{issue_code}.T" if issue_code and not issue_code.endswith(".T") else issue_code

            # Side mapping: "3"=BUY, "1"=SELL
            side_raw = str(pos.get("sOrderBaibaiKubun", ""))
            side = "BUY" if side_raw == "3" else "SELL"

            # Bensai mapping: neutral code mapping
            # Tachibana: 26=制度6ヶ月, 36=一般6ヶ月
            bensai_raw = str(pos.get("sOrderBensaiKubun", ""))
            neutral_margin = 1 if bensai_raw == "26" else 3

            # Account type mapping: "1"=特定, "3"=一般 -> neutral: 4=特定, 2=一般
            acc_raw = str(pos.get("sOrderZyoutoekiKazeiC", ""))
            neutral_acc = 4 if acc_raw == "1" else 2

            positions.append(
                Position(
                    ticker=ticker,
                    side=side,
                    quantity=int(pos.get("sOrderTategyokuSuryou", 0)),
                    price=float(pos.get("sOrderTategyokuTanka", 0.0)),
                    exchange=0, # Not applicable
                    execution_id=str(pos.get("sOrderTategyokuNumber", "")),
                    margin_trade_type=neutral_margin,
                    account_type=neutral_acc,
                    extra={
                        "sOrderHensaiKanouSuryou": int(pos.get("sOrderHensaiKanouSuryou", 0)),
                        "sOrderTategyokuDay": pos.get("sOrderTategyokuDay"),
                        "sOrderTategyokuKizituDay": pos.get("sOrderTategyokuKizituDay"),
                        "sOrderGaisanHyoukaSoneki": pos.get("sOrderGaisanHyoukaSoneki"),
                        "sOrderHyoukaTanka": pos.get("sOrderHyoukaTanka"),
                        "sOrderGaisanHyoukaSonekiRitu": pos.get("sOrderGaisanHyoukaSonekiRitu"),
                        "sOrderTategyokuDaikin": pos.get("sOrderTategyokuDaikin"),
                        "sOrderTateTesuryou": pos.get("sOrderTateTesuryou"),
                        "sOrderZyunHibu": pos.get("sOrderZyunHibu"),
                        "sOrderGyakuhibu": pos.get("sOrderGyakuhibu"),
                        "sOrderKasikaburyou": pos.get("sOrderKasikaburyou"),
                    },
                )
            )
        return positions

    def submit_order(
        self,
        order: OrderRequest,
        *,
        is_close: bool = False,
        close_position_order: int = 0,
    ) -> OrderResult:
        clean_ticker = order.ticker.replace(".T", "")

        # Side mapping: buy -> "3", sell -> "1"
        side_code = "3" if order.side == OrderSide.BUY else "1"

        # Margin Trade Type mapping
        # 1 = 制度信用 (Tachibana: 2=新規, 4=返済)
        # 2, 3 = 一般信用 (Tachibana: 6=新規, 8=返済)
        m_type = order.margin_trade_type if order.margin_trade_type is not None else self._broker_config.margin_trade_type
        if is_close:
            genkin_shinyou = "4" if m_type == 1 else "8"
        else:
            genkin_shinyou = "2" if m_type == 1 else "6"

        # Account Type mapping
        # 4 (特定) -> "1", 2 (一般) -> "3", 12 (法人) -> "9"
        acc_type = order.account_type if order.account_type is not None else self._broker_config.account_type
        account_code = "1"
        if acc_type == 2:
            account_code = "3"
        elif acc_type == 12:
            account_code = "9"

        # Order Type mapping
        # MARKET -> condition="0", price="0"
        # LIMIT -> condition="0", price=str(limit_price)
        # CLOSE -> condition="4" (引け), price="0"
        condition = "0"
        price_str = "0"

        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError("Limit price is required for limit orders")
            price_str = str(order.limit_price)
        elif order.order_type == OrderType.CLOSE:
            condition = "4"

        try:
            res = self._client.send_order(
                ticker=clean_ticker,
                side=side_code,
                quantity=order.quantity,
                order_price=price_str,
                condition=condition,
                genkin_shinyou=genkin_shinyou,
                account_type=account_code,
                is_close=is_close,
            )

            order_id = str(res.get("sOrderNumber", ""))
            return OrderResult(
                order_id=order_id,
                status=OrderStatus.SUBMITTED,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                margin_trade_type=self._broker_config.margin_trade_type,
            )

        except (TachibanaApiError, ValueError) as e:
            logger.error("Tachibana order failed for %s: %s", order.ticker, e)
            return OrderResult(
                order_id="",
                status=OrderStatus.FAILED,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                message=str(e),
            )

    def submit_orders_batch(
        self,
        orders: list[OrderRequest],
        *,
        delay_ms: int = 250,
        is_close: bool = False,
        close_position_order: int = 0,
    ) -> list[OrderResult]:
        """Place multiple orders with delay between each to avoid rate limits."""
        if not orders:
            return []

        logger.info(
            "[TachibanaBroker] Submitting %d batch orders (is_close=%s)...",
            len(orders),
            is_close,
        )

        results: list[OrderResult] = []
        for i, order in enumerate(orders):
            if i > 0:
                time.sleep(delay_ms / 1000.0)

            res = self.submit_order(
                order,
                is_close=is_close,
                close_position_order=close_position_order,
            )
            results.append(res)

        # --- Market-Neutrality Rollback Check ---
        # Skip rollback for close orders (返済) — they reduce positions, not open new ones.
        # Partial close failures are acceptable; the decision run will report them.
        if is_close:
            success = sum(1 for r in results if r.status == OrderStatus.SUBMITTED)
            logger.info("Batch close submission complete: %d/%d orders successful.", success, len(orders))
            return results

        buy_total = sum(1 for o in orders if o.side == OrderSide.BUY)
        sell_total = sum(1 for o in orders if o.side == OrderSide.SELL)
        buy_success = sum(1 for r in results if r.side == OrderSide.BUY and r.status == OrderStatus.SUBMITTED)
        sell_success = sum(1 for r in results if r.side == OrderSide.SELL and r.status == OrderStatus.SUBMITTED)
        buy_fail = buy_total - buy_success
        sell_fail = sell_total - sell_success

        needs_rollback = False
        if buy_total > 0 and buy_fail >= buy_total / 2:
            logger.error(
                "BUY side failure rate too high: %d/%d. Triggering rollback.",
                buy_fail,
                buy_total,
            )
            needs_rollback = True
        if sell_total > 0 and sell_fail >= sell_total / 2:
            logger.error(
                "SELL side failure rate too high: %d/%d. Triggering rollback.",
                sell_fail,
                sell_total,
            )
            needs_rollback = True

        if needs_rollback:
            logger.error("Rolling back %d successful orders due to imbalanced execution.", len(results))
            manual_interventions = []

            # Try to cancel successful orders
            # Since Tachibana cancel requires sEigyouDay, we get it from orders list or use current date
            current_date_str = datetime.now().strftime("%Y%m%d")

            for r in results:
                if r.status != OrderStatus.SUBMITTED or not r.order_id:
                    continue

                try:
                    # Cancel order
                    self._client.cancel_order(r.order_id, current_date_str)
                    logger.info("Cancelled order: %s", r.order_id)
                except Exception as e:
                    logger.error("Failed to cancel order %s: %s", r.order_id, e)
                    opposite_side = "SELL" if r.side == OrderSide.BUY else "BUY"
                    instruction = (
                        f"[Ticker: {r.ticker}] Cancel failed. Please check position. "
                        f"If executed, manually perform opposite trade: {opposite_side} {r.quantity} shares. "
                        f"(Order ID: {r.order_id})"
                    )
                    manual_interventions.append(instruction)

            if manual_interventions:
                logger.critical("================ 🚨 EMERGENCY: MANUAL RESTORATION REQUIRED 🚨 ================")
                logger.critical("Batch order execution failed to maintain market neutrality.")
                logger.critical("Please execute opposite trades manually for the following items:")
                for msg in manual_interventions:
                    logger.critical("  -> %s", msg)
                logger.critical("==========================================================================")
            else:
                logger.info("All successful orders in batch successfully cancelled. No manual action required.")

            return [
                OrderResult(
                    order_id=r.order_id,
                    status=OrderStatus.FAILED,
                    ticker=r.ticker,
                    side=r.side,
                    quantity=r.quantity,
                    message="Cancelled by market-neutrality rollback",
                )
                for r in results
            ]

        logger.info("Batch submission complete: %d/%d orders successful.", len(results), len(orders))
        return results
