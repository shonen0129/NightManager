import os
import sys
import logging
import argparse
from datetime import datetime

# Add src/ to the python import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from leadlag.broker.base import BrokerConfig
from leadlag.broker.factory import create_broker
from leadlag.core.types import OrderRequest, OrderSide, OrderType, OrderStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TachibanaDemoOrderTest")

def main():
    parser = argparse.ArgumentParser(description="Test Tachibana Securities Demo Order")
    parser.add_argument("--auth-id", help="Tachibana Auth ID (sAuthId)")
    parser.add_argument("--key-path", help="Path to e_api private key (.pem)")
    parser.add_argument("--password", help="Tachibana Second Password (trading password)")
    parser.add_argument("--ticker", default="1617.T", help="Ticker to buy (e.g. 1617.T)")
    parser.add_argument("--qty", type=int, default=100, help="Quantity to buy (default: 100)")
    parser.add_argument("--price", type=float, help="Limit price (if omitted, will query market price)")
    parser.add_argument("--market", action="store_true", help="Place a market order instead of limit")
    parser.add_argument("--env-file", help="Path to .env file to load")
    args = parser.parse_args()

    # Load dotenv if available
    try:
        import dotenv
        if args.env_file:
            dotenv.load_dotenv(args.env_file)
        else:
            # Try to load .env from project root
            root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
            if os.path.exists(root_env):
                logger.info(f"Loading environment variables from {root_env}")
                dotenv.load_dotenv(root_env)
    except ImportError:
        logger.warning("python-dotenv is not installed, relying on system environment variables")

    # Get credentials from args or env
    auth_id = args.auth_id or os.environ.get("TACHIBANA_AUTH_ID")
    key_path = args.key_path or os.environ.get("TACHIBANA_PRIVATE_KEY_PATH")
    password = args.password or os.environ.get("TACHIBANA_SECOND_PASSWORD")

    if not auth_id or not key_path or not password:
        logger.error("Missing required credentials!")
        print("\nPlease provide Tachibana Demo credentials using one of the following methods:")
        print("1. Command line arguments:")
        print("   python3 scripts/test_tachibana_demo_order.py --auth-id <id> --key-path <path_to_pem> --password <pwd>")
        print("2. Set environment variables or write a .env file in the project root containing:")
        print("   TACHIBANA_AUTH_ID=<your_auth_id>")
        print("   TACHIBANA_PRIVATE_KEY_PATH=<path_to_e_api_private_key.pem>")
        print("   TACHIBANA_SECOND_PASSWORD=<your_trading_password>")
        sys.exit(1)

    if not os.path.exists(key_path):
        logger.error(f"Private key file not found at path: {key_path}")
        sys.exit(1)

    logger.info("Initializing TachibanaBrokerClient on DEMO environment...")
    config = BrokerConfig(
        provider="tachibana",
        api_url="https://demo-kabuka.e-shiten.jp/e_api_v4r9",
        api_token=auth_id,
        api_password=password,
        request_timeout=15,
        margin_trade_type=3, # default general margin / day-trade
        account_type=4,      # Specific/特定
        extra={"private_key_path": key_path}
    )

    try:
        client = create_broker(config)
    except Exception as e:
        logger.exception("Failed to instantiate broker client")
        sys.exit(1)

    try:
        logger.info("Running health check (attempts login)...")
        # Direct login call for verification
        client._client.login()
        logger.info("Login successful!")

        logger.info("Retrieving wallet balance...")
        wallet = client.get_wallet()
        print("\n--- Wallet Summary ---")
        print(f"Cash Available (Genbutsu): {wallet.cash_available:,.2f} JPY")
        print(f"Margin Available (Shinyou): {wallet.margin_available:,.2f} JPY")
        print(f"Extra details: {wallet.extra}")
        print("----------------------\n")

        logger.info("Retrieving open positions...")
        positions = client.get_positions()
        print(f"Open credit positions count: {len(positions)}")
        for i, pos in enumerate(positions):
            print(f"  [{i}] Ticker: {pos.ticker}, Side: {pos.side}, Qty: {pos.quantity}, Price: {pos.price}")

        # Fetch current price for limit order if price not specified
        limit_price = args.price
        if not args.market and limit_price is None:
            logger.info(f"Fetching today's open price for {args.ticker} to set limit price...")
            try:
                opens = client.fetch_open_prices([args.ticker], allow_missing=True)
                current_open = opens.get(args.ticker)
                if current_open and current_open > 0:
                    # Set limit price slightly lower than open price to be safe
                    limit_price = round(current_open * 0.95, 1)
                    logger.info(f"Today's open price is {current_open}. Setting limit price to 95% of open: {limit_price}")
                else:
                    # Default backup price
                    limit_price = 1500.0
                    logger.warning(f"Could not get open price for {args.ticker}. Using default limit price: {limit_price}")
            except Exception as e:
                limit_price = 1500.0
                logger.warning(f"Failed to query price: {e}. Using default limit price: {limit_price}")

        # Construct order request
        order_type = OrderType.MARKET if args.market else OrderType.LIMIT
        order_req = OrderRequest(
            ticker=args.ticker,
            side=OrderSide.BUY,
            quantity=args.qty,
            order_type=order_type,
            limit_price=limit_price if order_type == OrderType.LIMIT else None
        )

        order_desc = f"{order_req.side.name} {order_req.quantity} shares of {order_req.ticker} via {order_req.order_type.name}"
        if order_req.limit_price:
            order_desc += f" @ {order_req.limit_price}"

        print(f"\nPlacing order: {order_desc}")
        confirm = input("Do you want to proceed with this demo order? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Order cancelled by user.")
            client.close()
            return

        logger.info("Submitting order...")
        result = client.submit_order(order_req)

        print("\n--- Order Submission Result ---")
        print(f"Status: {result.status.name}")
        print(f"Order ID: {result.order_id}")
        print(f"Message/Error: {result.message}")
        print("--------------------------------\n")

        if result.status == OrderStatus.SUBMITTED:
            # Query order list to verify it's there
            logger.info("Querying active orders...")
            orders = client._client.get_order_list()
            print(f"Today's orders list (count={len(orders)}):")
            for ord_data in orders:
                ord_num = ord_data.get("sOrderNumber")
                if ord_num == result.order_id:
                    print(f"  -> FOUND SUBMITTED ORDER: ID={ord_num}, Ticker={ord_data.get('sIssueCode')}, Baibai={ord_data.get('sBaibaiKubun')}, Price={ord_data.get('sOrderPrice')}, Status={ord_data.get('sResultText') or ord_data.get('sResultCode')}")
                else:
                    print(f"  Order ID={ord_num}, Ticker={ord_data.get('sIssueCode')}, Price={ord_data.get('sOrderPrice')}")

            # Ask if user wants to cancel the order
            cancel_confirm = input(f"Do you want to cancel the submitted order {result.order_id}? (y/N): ").strip().lower()
            if cancel_confirm == 'y':
                logger.info(f"Cancelling order {result.order_id}...")
                current_date_str = datetime.now().strftime("%Y%m%d")
                cancel_res = client._client.cancel_order(result.order_id, current_date_str)
                print(f"Cancel API response code: {cancel_res.get('sResultCode')}, text: {cancel_res.get('sResultText')}")

    except Exception as e:
        logger.exception("An error occurred during demo execution")
    finally:
        client.close()

if __name__ == "__main__":
    main()
