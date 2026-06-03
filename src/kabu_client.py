"""
kabuステーション API クライアント
株価情報の取得と売買注文実行機能を提供する
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import requests

logger = logging.getLogger(__name__)

# 401 Unauthorized handling
MAX_401_RETRIES = 3

# リトライ設定
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 1.0
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def issue_api_token(
    api_url: str,
    api_password: str,
    request_timeout: int = 10,
) -> str:
    """Issue a new API token via /token endpoint."""
    if not api_password:
        raise ValueError("KABU_API_PASSWORD is empty")

    token_url = f"{api_url}/token"
    payload = {"APIPassword": api_password}
    response = requests.post(
        token_url,
        json=payload,
        timeout=request_timeout,
    )
    response.raise_for_status()

    result = response.json()
    token = result.get("Token")
    if not token:
        raise ValueError("Token issuance failed: no Token in response")

    return str(token)


@dataclass
class KabuConfig:
    """kabuステーション API設定"""

    api_url: str  # e.g., "http://localhost:18080"
    api_token: str  # APIトークン
    request_timeout: int = 10  # request timeout (seconds)


class KabuApiError(Exception):
    """Raised when kabuステーション API interactions fail."""

    def __init__(
        self,
        message: str,
        *,
        endpoint: Optional[str] = None,
        ticker: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self.ticker = ticker
        super().__init__(message)


class KabuClient:
    """
    kabuステーション API クライアント

    Attributes:
        config: KabuConfig インスタンス
        session: requests.Session インスタンス

    Usage:
        # Context manager の使用を推奨
        with KabuClient(config) as client:
            client.send_order(...)

        # 明示的なクローズ
        client = KabuClient(config)
        try:
            client.send_order(...)
        finally:
            client.close()
    """

    def __init__(self, config: KabuConfig):
        """
        Initialize KabuClient

        Args:
            config: KabuConfig インスタンス
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-API-KEY": config.api_token,
                "Content-Type": "application/json",
            }
        )

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: セッションをクローズ"""
        self.close()
        return False

    def close(self):
        """セッションをクローズし、接続プールを解放"""
        if self.session is not None:
            self.session.close()
            self.session = None
            logger.debug("HTTP session closed and resources released")

    def _refresh_token(self) -> bool:
        """Attempt to refresh the API token by issuing a new one.

        Reads APIPassword from environment variable KABU_API_PASSWORD
        and calls the /token endpoint to get a new token.

        Returns:
            True if token was refreshed successfully
        """
        api_password = os.environ.get("KABU_API_PASSWORD", "")
        if not api_password:
            logger.error(
                "Cannot refresh token: KABU_API_PASSWORD environment variable is not set"
            )
            return False

        try:
            new_token = issue_api_token(
                self.config.api_url,
                api_password,
                request_timeout=self.config.request_timeout,
            )

            self.config = KabuConfig(
                api_url=self.config.api_url,
                api_token=new_token,
                request_timeout=self.config.request_timeout,
            )
            self.session.headers.update(
                {
                    "X-API-KEY": new_token,
                    "Content-Type": "application/json",
                }
            )
            os.environ["KABU_API_TOKEN"] = new_token
            logger.info("API token refreshed successfully")
            return True
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        allow_token_refresh: bool = True,
    ) -> Dict[str, Any]:
        """
        Internal method to make HTTP requests to kabuステーション API
        リトライ機能付き（exponential backoff）
        401 Unauthorized 時はトークン再発行を試みる。

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/positions")
            data: Request body (for POST/PUT) or query params (for GET)
            max_retries: 最大リトライ回数
            backoff_factor: リトライ待機時間の係数（秒）
            allow_token_refresh: 401時にトークン再発行を試みるか

        Returns:
            API response as dict

        Raises:
            requests.RequestException: Network or API error (after retries)
            ValueError: Invalid response
        """
        url = f"{self.config.api_url}{endpoint}"
        last_exception: Optional[Exception] = None

        def _summarize_error_body(resp: requests.Response) -> str:
            try:
                body = resp.text or ""
            except Exception:
                return "<unreadable body>"
            if not body:
                return "<empty body>"
            body = " ".join(body.split())
            if len(body) > 500:
                body = body[:500] + "...(truncated)"
            return body

        for attempt in range(max_retries):
            try:
                if method == "GET":
                    response = self.session.get(
                        url,
                        params=data,  # GET uses params for query string
                        timeout=self.config.request_timeout,
                    )
                elif method == "POST":
                    response = self.session.post(
                        url,
                        json=data,
                        timeout=self.config.request_timeout,
                    )
                elif method == "PUT":
                    response = self.session.put(
                        url,
                        json=data,
                        timeout=self.config.request_timeout,
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # 401 Unauthorized: token may have expired
                if response.status_code == 401 and allow_token_refresh:
                    if endpoint == "/token":
                        # Don't retry token endpoint itself
                        response.raise_for_status()

                    logger.warning(
                        "Received 401 Unauthorized. Attempting token refresh..."
                    )
                    if self._refresh_token():
                        # Retry with new token (only once for token refresh)
                        response = (
                            self.session.get(
                                url,
                                params=data,
                                timeout=self.config.request_timeout,
                            )
                            if method == "GET"
                            else (
                                self.session.post(
                                    url,
                                    json=data,
                                    timeout=self.config.request_timeout,
                                )
                                if method == "POST"
                                else self.session.put(
                                    url,
                                    json=data,
                                    timeout=self.config.request_timeout,
                                )
                            )
                        )
                        if response.status_code == 200:
                            return response.json()

                # リトライ可能なステータスコードの場合、待機して再試行
                if response.status_code in RETRYABLE_STATUS_CODES:
                    last_exception = requests.RequestException(
                        f"Server returned {response.status_code}: {response.text}"
                    )
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor * (2**attempt)
                        logger.warning(
                            f"API request returned {response.status_code}. "
                            f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    # 最後の試行でも失敗した場合、例外を送出
                    logger.error(
                        "API error response: %s %s -> %s %s",
                        method,
                        endpoint,
                        response.status_code,
                        _summarize_error_body(response),
                    )
                    response.raise_for_status()

                if response.status_code >= 400 and attempt >= max_retries - 1:
                    logger.error(
                        "API error response: %s %s -> %s %s",
                        method,
                        endpoint,
                        response.status_code,
                        _summarize_error_body(response),
                    )

                response.raise_for_status()
                return response.json()

            except requests.RequestException as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = backoff_factor * (2**attempt)
                    logger.warning(
                        f"API request failed: {method} {endpoint}: {e}. "
                        f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"API request failed after {max_retries} attempts: "
                        f"{method} {endpoint}: {e}"
                    )
                    raise

        # Should not reach here, but just in case
        raise last_exception or requests.RequestException(
            f"Unexpected error in API request: {method} {endpoint}"
        )

    def health_check(self, symbol: str = "1617@1") -> bool:
        """
        Check API connectivity and authentication

        Args:
            symbol: Symbol to test with (default: "1617@1" = ETF on 東証)

        Returns:
            True if API is accessible and authenticated
        """
        try:
            result = self._request("GET", f"/board/{symbol}")
            # API returns BoardSuccess response directly (not nested under "result")
            is_valid = isinstance(result, dict) and "Symbol" in result
            if is_valid:
                logger.info("API health check passed")
            else:
                logger.warning("API health check failed: invalid response format")
            return is_valid
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False

    def get_margin_wallet(self) -> Dict[str, float]:
        """Get margin trading wallet info (信用取引余力).

        Calls GET /wallet/margin to retrieve margin account balance.

        Returns:
            Dict with keys:
                - margin_available: 信用新規可能額 (MarginAccountWallet)

        Raises:
            ValueError: If API returns invalid data or margin info unavailable
        """
        try:
            result = self._request("GET", "/wallet/margin")

            margin_available = result.get("MarginAccountWallet")
            if margin_available is None:
                raise ValueError(
                    "MarginAccountWallet not found in API response. "
                    "Verify margin trading is enabled for this account."
                )

            return {
                "margin_available": float(margin_available),
                "deposit_keep_rate": float(result.get("DepositkeepRate", 0.0)),
            }

        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch margin wallet: {e}")

    def get_cash_wallet(self) -> Dict[str, float]:
        """Get cash trading wallet info (現物取引余力).

        Calls GET /wallet/cash to retrieve cash account balance.

        Returns:
            Dict with keys:
                - cash_available: 現物買付可能額 (StockAccountWallet)
                - au_kc_cash_available: 三菱UFJ eスマート証券可能額 (AuKCStockAccountWallet)
                - au_jbn_cash_available: auじぶん銀行残高 (AuJbnStockAccountWallet)

        Raises:
            ValueError: If API returns invalid data or cash info unavailable
        """
        try:
            result = self._request("GET", "/wallet/cash")

            cash_available = result.get("StockAccountWallet")
            au_kc_available = result.get("AuKCStockAccountWallet")
            au_jbn_available = result.get("AuJbnStockAccountWallet")
            if cash_available is None:
                cash_available = au_kc_available

            if cash_available is None:
                raise ValueError(
                    "StockAccountWallet not found in API response. "
                    "Verify cash trading is enabled for this account."
                )

            return {
                "cash_available": float(cash_available),
                "au_kc_cash_available": float(au_kc_available)
                if au_kc_available is not None
                else 0.0,
                "au_jbn_cash_available": float(au_jbn_available)
                if au_jbn_available is not None
                else 0.0,
            }

        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch cash wallet: {e}")

    def get_price(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Get current stock price and related info

        Args:
            ticker: Stock ticker symbol in kabuステーション format
                    (e.g., "1617@1" for ETF on 東証)

        Returns:
            Dict with keys: bid, ask, last, open, high, low, volume
            or None if ticker not found
        """
        try:
            # kabuステーション API では /board/{symbol} でリアルタイム価格を取得
            result = self._request("GET", f"/board/{ticker}")

            # API returns BoardSuccess response directly (not nested)
            # Check for error via StatusCode if present
            if result.get("StatusCode", 0) != 0:
                logger.error(
                    f"Error fetching price for {ticker}: {result.get('Message', 'Unknown error')}"
                )
                return None

            # Response is the data itself per API spec (no nesting under "data" key)
            # Per kabu STATION API spec:
            #   CurrentPrice (現値): After US market close, this is the US closing price
            #   PreviousClose (前日終値): Previous day's closing price
            #   OpeningPrice (始値): Today's opening price
            prev_close_raw = result.get("PreviousClose")
            current_price_raw = result.get("CurrentPrice")
            return {
                "bid": float(result.get("BidPrice", 0)),
                "ask": float(result.get("AskPrice", 0)),
                "last": (
                    float(current_price_raw) if current_price_raw is not None else 0
                ),
                "open": float(result.get("OpeningPrice", 0)),
                "high": float(result.get("HighPrice", 0)),
                "low": float(result.get("LowPrice", 0)),
                "volume": int(result.get("TradingVolume", 0)),
                # For close-to-close return computation (used by fetch_us_etf_returns)
                "PreviousClose": (
                    float(prev_close_raw) if prev_close_raw is not None else None
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get price for {ticker}: {e}")
            return None

    @staticmethod
    def _to_kabu_symbol(yf_ticker: str) -> str:
        """Convert yfinance ticker to kabuステーション format.

        Examples:
            '1617.T' -> '1617@1'   (東証)
        """
        code = yf_ticker.replace(".T", "")
        return f"{code}@1"

    def fetch_us_etf_returns(
        self,
        us_tickers: List[str],
        delay_ms: int = 50,
        parallel: bool = True,
        max_workers: int = 5,
    ) -> Dict[str, float]:
        """Fetch US ETF close-to-close returns via kabuステーション /board API.

        After US market close (16:00 ET / ~5:00-6:00 JST next day), the API's
        CurrentPrice reflects the US closing price. We compute close-to-close
        returns using the formula:

            return = (CurrentPrice - PreviousClose) / PreviousClose

        Where:
            CurrentPrice (現値): Current price, which after market close = closing price
            PreviousClose (前日終値): Previous trading day's closing price

        This provides accurate close-to-close returns without approximation.

        Timing note: This should be called after US market close to get the
        most recent closing prices. For US ETFs, PreviousClose represents
        the close from two US sessions ago, and CurrentPrice represents
        the close from the most recent US session.

        Args:
            us_tickers: List of US ETF ticker symbols (e.g., ['XLB', 'XLC', ...])
            delay_ms: Delay between API calls in milliseconds
            parallel: If True, use ThreadPoolExecutor for parallel fetching
            max_workers: Maximum number of parallel threads

        Returns:
            Dict mapping US ETF ticker -> close-to-close return
        """
        logger.info(
            f"Fetching US ETF close-to-close returns for {len(us_tickers)} tickers..."
        )

        def _fetch_single(ticker: str) -> tuple:
            # US tickers in kabu format: XLB@31 (NYSE ARCA)
            kabu_sym = f"{ticker}@31"
            price_info = self.get_price(kabu_sym)
            if price_info is not None:
                # Per kabu STATION API spec:
                # CurrentPrice becomes the closing price after market close
                # PreviousClose is the previous day's closing price
                # Return = (CurrentPrice - PreviousClose) / PreviousClose
                current_price = price_info.get("last", 0)  # CurrentPrice
                prev_close = price_info.get("PreviousClose")

                if prev_close is not None and prev_close > 0 and current_price > 0:
                    return_price = current_price / prev_close - 1.0
                    logger.debug(
                        f"{ticker}: CurrentPrice={current_price}, "
                        f"PreviousClose={prev_close}, return={return_price:.6f}"
                    )
                    return (ticker, return_price)

                logger.warning(
                    f"Failed to compute return for {ticker}: "
                    f"CurrentPrice={current_price}, PreviousClose={prev_close}"
                )
            return (ticker, None)

        returns: Dict[str, float] = {}
        failed: List[str] = []

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_fetch_single, tk): tk for tk in us_tickers}
                for future in as_completed(futures):
                    ticker, ret = future.result()
                    if ret is not None:
                        returns[ticker] = ret
                    else:
                        failed.append(ticker)
        else:
            for i, ticker in enumerate(us_tickers):
                if i > 0:
                    time.sleep(delay_ms / 1000.0)
                ticker_sym, ret = _fetch_single(ticker)
                if ret is not None:
                    returns[ticker_sym] = ret
                else:
                    failed.append(ticker_sym)

        if failed:
            logger.warning(
                f"Failed to fetch returns for {len(failed)} US ETF(s): "
                f"{', '.join(failed)}"
            )

        logger.info(
            f"Successfully fetched returns for {len(returns)}/{len(us_tickers)} US ETFs"
        )
        return returns

    def fetch_jp_open_prices(
        self,
        jp_tickers: List[str],
        delay_ms: int = 50,
        parallel: bool = True,
        max_workers: int = 5,
        allow_missing: bool = False,
    ) -> Dict[str, float]:
        """Fetch today's open prices for JP tickers via kabuステーション /board API.

        Args:
            jp_tickers: List of yfinance-style tickers (e.g., ['1617.T', ...])
            delay_ms: Delay between API calls in milliseconds (used in sequential mode)
            parallel: If True, use ThreadPoolExecutor for parallel fetching
            max_workers: Maximum number of parallel threads (only used when parallel=True)

        Returns:
            Dict mapping yfinance ticker -> open price

        Raises:
            ValueError: If any ticker fails to return a valid open price
                (unless allow_missing=True)
        """
        opens: Dict[str, float] = {}
        failed: List[str] = []

        logger.info(
            f"Fetching open prices for {len(jp_tickers)} tickers (parallel={parallel})..."
        )

        if parallel:
            # Parallel fetching using ThreadPoolExecutor
            def _fetch_single(ticker: str) -> tuple:
                kabu_sym = self._to_kabu_symbol(ticker)
                price_info = self.get_price(kabu_sym)
                if price_info is not None and price_info["open"] > 0:
                    return (ticker, price_info["open"])
                else:
                    return (ticker, None)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_fetch_single, tk): tk for tk in jp_tickers}
                for future in as_completed(futures):
                    ticker, price = future.result()
                    if price is not None:
                        opens[ticker] = price
                    else:
                        failed.append(ticker)
        else:
            # Sequential fetching (original behavior)
            for i, ticker in enumerate(jp_tickers):
                if i > 0:
                    time.sleep(delay_ms / 1000.0)

                kabu_sym = self._to_kabu_symbol(ticker)
                price_info = self.get_price(kabu_sym)

                if price_info is not None and price_info["open"] > 0:
                    opens[ticker] = price_info["open"]
                else:
                    failed.append(ticker)

        if failed:
            message = (
                f"Failed to fetch open prices for {len(failed)} ticker(s): "
                f"{', '.join(failed)}"
            )
            if allow_missing:
                logger.warning(message)
            else:
                logger.error(message)
                raise ValueError(message)

        logger.info(f"Successfully fetched open prices for {len(opens)} tickers")
        return opens

    def get_positions(
        self,
        product: int = 0,
        symbol: Optional[str] = None,
        side: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get current open positions via GET /positions.

        Per kabu STATION API spec:
        - product: 0=all, 1=cash, 2=margin, 3=futures, 4=options
        - symbol: optional ticker@exchange filter
        - side: 1=sell, 2=buy

        Returns:
            List of position dicts aligned with API response fields:
            Symbol, SymbolName, Exchange, ExecutionDay, Price,
            LeavesQty, HoldQty, Side, etc.

        Raises:
            KabuApiError: If API call fails or response format is invalid
        """
        params: Dict[str, Any] = {"product": str(product)}
        if symbol is not None:
            params["symbol"] = symbol
        if side is not None:
            params["side"] = str(side)

        try:
            positions = self._request("GET", "/positions", data=params)

            if not isinstance(positions, list):
                raise KabuApiError(
                    f"Unexpected positions response type: {type(positions)}",
                    endpoint="/positions",
                )

            mapped = []
            for pos in positions:
                mapped.append(
                    {
                        "ticker": str(pos.get("Symbol", "")),
                        "symbol_name": str(pos.get("SymbolName", "")),
                        "exchange": int(pos.get("Exchange", 0)),
                        "execution_id": str(pos.get("ExecutionID", "")),
                        "account_type": int(pos.get("AccountType", 0)),
                        "execution_day": pos.get("ExecutionDay"),
                        "side": "BUY" if str(pos.get("Side")) == "2" else "SELL",
                        "side_raw": str(pos.get("Side", "")),
                        "quantity": int(pos.get("LeavesQty", 0)),
                        "leaves_qty": int(pos.get("LeavesQty", 0)),
                        "hold_qty": int(pos.get("HoldQty", 0)),
                        "price": float(pos.get("Price", 0)),
                        "margin_trade_type": pos.get("MarginTradeType"),
                        "expire_day": pos.get("ExpireDay"),
                        "current_price": pos.get("CurrentPrice"),
                        "valuation": pos.get("Valuation"),
                        "profit_loss": pos.get("ProfitLoss"),
                        "profit_loss_rate": pos.get("ProfitLossRate"),
                    }
                )

            logger.info(f"Retrieved {len(mapped)} open position(s)")
            return mapped

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise KabuApiError(
                f"Failed to fetch positions: {e}",
                endpoint="/positions",
            ) from e

    def send_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        exchange: int = 27,
        order_type: str = "MO",  # MO=MarketOrder, LO=LimitOrder
        limit_price: Optional[float] = None,
        margin_trade_type: int = 3,  # 1=制度, 2=一般(長期), 3=一般(デイトレ)
        account_type: int = 4,  # 2=一般, 4=特定, 12=法人
        is_close: bool = False,  # True for position closing (信用返済)
        close_position_order: int = 0,  # Close order priority (0-7) for credit repayment
    ) -> Dict[str, Any]:
        """
        Send a margin (信用) buy/sell order

        Args:
            is_close: If True, send as position closing order (信用返済).
                      If False, send as new position order (信用新規).
            close_position_order: Close order priority (0-7) for credit repayment.

        Args:
            ticker: Stock ticker symbol (e.g., "1617" without exchange suffix)
            side: "BUY" or "SELL"
            quantity: Number of shares
            exchange: Market code (default: 27 = 東証+)
            order_type: "MO" (market order) or "LO" (limit order)
            limit_price: Limit price (required if order_type=="LO")
            margin_trade_type: 1=制度信用, 2=一般信用(長期), 3=一般信用(デイトレ)
            account_type: 2=一般口座, 4=特定口座, 12=法人口座

        Returns:
            Order response with keys: order_id, status, ticker, side, quantity

        Raises:
            ValueError: Invalid order input
            KabuApiError: API call failure or API-side order rejection
        """
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}")

        if side not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid side: {side}")

        if order_type == "LO" and limit_price is None:
            raise ValueError("Limit price required for limit order")

        if is_close:
            if close_position_order is None:
                raise ValueError("close_position_order is required for close orders")
            close_position_order = int(close_position_order)
            if not 0 <= close_position_order <= 7:
                raise ValueError("close_position_order must be between 0 and 7")

        # Map order_type to FrontOrderType per kabu STATION API spec:
        # 10=成行, 15=引成（前場）, 16=引成（後場）, 20=指値
        # "MO"=Market Order (成行), "LO"=Limit Order (指値),
        # "CLO"=Close Order (引成後場 - end-of-day at-to-pita closing)
        if order_type == "MO":
            front_order_type = 10  # 成行
            price = 0
        elif order_type == "CLO":
            front_order_type = 16  # 引成（後場）
            price = 0
        elif order_type == "LO":
            front_order_type = 20  # 指値
            price = limit_price if limit_price is not None else 0
        else:
            raise ValueError(f"Unknown order_type: {order_type}")

        try:
            if is_close:
                # 信用返済注文 (CashMargin=3)
                payload = {
                    "Symbol": ticker,
                    "Exchange": exchange,
                    "SecurityType": 1,  # 株式
                    "Side": "2" if side == "BUY" else "1",  # API: 1=売, 2=買
                    "CashMargin": 3,  # 信用返済
                    "MarginTradeType": margin_trade_type,
                    "DelivType": 0,  # 信用返済は指定なし
                    "FundType": "11",  # 信用取引
                    "AccountType": account_type,
                    "Qty": quantity,
                    "ClosePositionOrder": close_position_order,
                    "FrontOrderType": front_order_type,
                    "Price": price,
                    "ExpireDay": 0,  # 当日
                }
            else:
                # 信用新規注文 (CashMargin=2)
                payload = {
                    "Symbol": ticker,
                    "Exchange": exchange,
                    "SecurityType": 1,  # 株式
                    "Side": "2" if side == "BUY" else "1",  # API: 1=売, 2=買
                    "CashMargin": 2,  # 信用新規
                    "MarginTradeType": margin_trade_type,
                    "DelivType": 0,  # 信用新規は指定なし
                    "FundType": "11",  # 信用取引
                    "AccountType": account_type,
                    "Qty": quantity,
                    "FrontOrderType": front_order_type,
                    "Price": price,
                    "ExpireDay": 0,  # 当日
                }

            result = self._request("POST", "/sendorder", data=payload)

            if result.get("Result") != 0:
                raise KabuApiError(
                    f"Order failed for {ticker}: {result.get('Message', result)}",
                    endpoint="/sendorder",
                    ticker=ticker,
                )

            order_response = {
                "order_id": str(result.get("OrderId", "")),
                "status": "SUBMITTED",
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "limit_price": limit_price,
                "margin_trade_type": margin_trade_type,
            }

            logger.info(
                f"Order submitted: {side} {quantity} shares of {ticker} "
                f"(Order ID: {order_response['order_id']})"
            )
            return order_response

        except ValueError:
            raise
        except KabuApiError:
            raise
        except Exception as e:
            logger.error(f"Failed to send order for {ticker}: {e}")
            raise KabuApiError(
                f"Failed to send order for {ticker}: {e}",
                endpoint="/sendorder",
                ticker=ticker,
            ) from e

    def get_order_status(
        self,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        product: int = 0,
        details: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get order status via GET /orders.

        Per kabu STATION API spec, orders are queried via query parameters:
        - id: specific order ID
        - product: 0=all, 1=cash, 2=margin, 3=futures, 4=options
        - details: include order details

        Returns:
            List of order status dicts. Empty list if no orders found.

        Raises:
            KabuApiError: If API call fails or response format is invalid
        """
        params: Dict[str, Any] = {
            "product": str(product),
            "details": str(details).lower(),
        }
        if order_id is not None:
            params["id"] = order_id
        if symbol is not None:
            params["symbol"] = symbol

        try:
            orders = self._request("GET", "/orders", data=params)

            if not isinstance(orders, list):
                raise KabuApiError(
                    f"Unexpected orders response type: {type(orders)}",
                    endpoint="/orders",
                )

            mapped = []
            for order in orders:
                mapped.append(
                    {
                        "order_id": str(order.get("ID", "")),
                        "state": int(order.get("State", 0)),
                        "symbol": str(order.get("Symbol", "")),
                        "side": "BUY" if str(order.get("Side")) == "2" else "SELL",
                        "order_qty": order.get("OrderQty", 0),
                        "cum_qty": order.get("CumQty", 0),
                        "price": order.get("Price"),
                        "cash_margin": order.get("CashMargin"),
                        "margin_trade_type": order.get("MarginTradeType"),
                        "expire_day": order.get("ExpireDay"),
                        "details": order.get("Details", []),
                    }
                )

            return mapped

        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            raise KabuApiError(
                f"Failed to get order status: {e}",
                endpoint="/orders",
            ) from e

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order via PUT /cancelorder.

        Per kabu STATION API spec:
        - Endpoint: PUT /cancelorder
        - Request body: {"OrderId": "xxx"}
        """
        try:
            result = self._request("PUT", "/cancelorder", data={"OrderId": order_id})

            # Response uses "Result" field (0 = success)
            if result.get("Result", -1) != 0:
                logger.error(
                    f"Cancel failed for {order_id}: " f"{result.get('Message', result)}"
                )
                return False

            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def place_orders_batch(
        self,
        orders: List[Dict[str, Any]],
        delay_ms: int = 50,
        margin_trade_type: int = 3,
        account_type: int = 4,
        is_close: bool = False,
        close_position_order: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Place multiple margin orders with delay between each to avoid rate limiting.

        Args:
            orders: List of order dicts with keys:
                    ticker, side, quantity, (order_type, limit_price optional)
            delay_ms: Delay between orders in milliseconds
            margin_trade_type: 1=制度信用, 2=一般信用(長期), 3=一般信用(デイトレ)
            account_type: 2=一般口座, 4=特定口座, 12=法人口座
            is_close: If True, send as position closing orders (信用返済)
            close_position_order: Close order priority (0-7) for credit repayment.
        Returns:
            List of order responses from send_order
        """
        order_type_label = "CLOSE" if is_close else "NEW"
        logger.info(
            f"Starting batch order submission ({order_type_label}): "
            f"{len(orders)} orders, delay={delay_ms}ms"
        )

        results = []
        failed = []
        for i, order in enumerate(orders):
            if i > 0:
                time.sleep(delay_ms / 1000.0)

            try:
                result = self.send_order(
                    ticker=order["ticker"],
                    side=order["side"],
                    quantity=order["quantity"],
                    exchange=order.get("exchange", 27),
                    order_type=order.get("order_type", "MO"),
                    limit_price=order.get("limit_price"),
                    margin_trade_type=order.get("margin_trade_type", margin_trade_type),
                    account_type=order.get("account_type", account_type),
                    is_close=is_close,
                    close_position_order=close_position_order,
                )
                results.append(result)
            except Exception as e:
                failed.append(order)
                logger.error(
                    "Order failed for %s (%s x%s): %s",
                    order["ticker"],
                    order["side"],
                    order["quantity"],
                    e,
                )

        # --- Market-neutrality rollback check ---
        buy_total = sum(1 for o in orders if o["side"] == "BUY")
        sell_total = sum(1 for o in orders if o["side"] == "SELL")
        buy_success = sum(1 for r in results if r["side"] == "BUY")
        sell_success = sum(1 for r in results if r["side"] == "SELL")
        buy_fail = buy_total - buy_success
        sell_fail = sell_total - sell_success

        needs_rollback = False
        if buy_total > 0 and buy_fail > buy_total / 2:
            logger.error(
                f"BUY side failure rate too high: {buy_fail}/{buy_total}. "
                "Triggering rollback to preserve market neutrality."
            )
            needs_rollback = True
        if sell_total > 0 and sell_fail > sell_total / 2:
            logger.error(
                f"SELL side failure rate too high: {sell_fail}/{sell_total}. "
                "Triggering rollback to preserve market neutrality."
            )
            needs_rollback = True

        if needs_rollback:
            logger.error(
                f"Rolling back {len(results)} successful orders "
                "due to imbalanced execution."
            )
            manual_interventions = []
            for r in results:
                order_id = r.get("order_id", "")
                ticker = r.get("ticker", "")
                side = r.get("side", "")
                quantity = r.get("quantity", 0)

                if order_id:
                    # ステータス（約定数量）を確認
                    try:
                        status_list = self.get_order_status(order_id)
                    except Exception as e:
                        logger.error(
                            "Failed to get order status during rollback for %s: %s",
                            order_id,
                            e,
                        )
                        opposite_side = "SELL" if side == "BUY" else "BUY"
                        instruction = (
                            f"[ティッカー: {ticker}] 注文状態取得に失敗しました。"
                            f"手動でポジション有無を確認し、存在する場合は"
                            f"【{opposite_side} {quantity} 口】の決済を行ってください。"
                            f"(元注文: {side}, ID: {order_id})"
                        )
                        manual_interventions.append(instruction)
                        continue

                    # get_order_status returns a list; extract cum_qty from first match
                    filled_qty = 0
                    if status_list and len(status_list) > 0:
                        filled_qty = status_list[0].get("cum_qty", 0)

                    if filled_qty > 0:
                        opposite_side = "SELL" if side == "BUY" else "BUY"
                        instruction = (
                            f"[ティッカー: {ticker}] 既に {filled_qty} 口 約定済です。"
                            f"手動で【{opposite_side} {filled_qty} 口】の発注（反対売買）を行い、"
                            f"ポジションを解消して下さい。(元注文: {side}, ID: {order_id})"
                        )
                        manual_interventions.append(instruction)
                    else:
                        # まだ未約定に見える場合はキャンセルを試行
                        cancelled = self.cancel_order(order_id)
                        if not cancelled:
                            opposite_side = "SELL" if side == "BUY" else "BUY"
                            instruction = (
                                f"[ティッカー: {ticker}] 未約定注文のキャンセルに失敗しました。"
                                f"すれ違いで約定した可能性があります。手動でポジションの有無を確認し、"
                                f"存在する場合は【{opposite_side} {quantity} 口】の決済を行ってください。"
                                f"(元注文: {side}, ID: {order_id})"
                            )
                            manual_interventions.append(instruction)

            if manual_interventions:
                logger.critical(
                    "================ 🚨 緊急：手動修復指示 🚨 ================"
                )
                logger.critical(
                    "片側のバッチ注文が過半数失敗したため、市場中立性が崩れました。"
                )
                logger.critical(
                    "以下の銘柄について、手動で反対売買を行い市場中立を回復して下さい："
                )
                for msg in manual_interventions:
                    logger.critical("  -> " + msg)
                logger.critical(
                    "=========================================================="
                )
            else:
                logger.info(
                    "全ての成功注文のキャンセル（ロールバック）に成功しました。手動対応は不要です。"
                )

            return []

        if failed:
            logger.warning(
                f"{len(failed)}/{len(orders)} orders failed. "
                f"Failed tickers: {[o['ticker'] for o in failed]}"
            )
        else:
            logger.info(
                f"Batch order completed: {len(results)}/{len(orders)} successful"
            )

        return results
