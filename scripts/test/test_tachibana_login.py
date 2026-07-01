import os
import sys
import logging

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from leadlag.broker.base import BrokerConfig
from leadlag.broker.factory import create_broker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TachibanaDemoLoginTest")

def main():
    # Load dotenv from project root
    try:
        import dotenv
        root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
        if os.path.exists(root_env):
            logger.info(f"Loading environment variables from {root_env}")
            dotenv.load_dotenv(root_env)
        else:
            logger.warning(f".env file not found at {root_env}")
    except ImportError:
        logger.warning("python-dotenv is not installed, relying on system environment variables")

    auth_id = os.environ.get("TACHIBANA_AUTH_ID")
    key_path = os.environ.get("TACHIBANA_PRIVATE_KEY_PATH")
    password = os.environ.get("TACHIBANA_SECOND_PASSWORD")
    api_url = os.environ.get("TACHIBANA_API_URL", "https://demo-kabuka.e-shiten.jp/e_api_v4r9")

    if not auth_id or not key_path or not password:
        logger.error("Missing required credentials in environment variables!")
        logger.info(f"Loaded credentials status: TACHIBANA_AUTH_ID={'SET' if auth_id else 'NOT SET'}, "
                    f"TACHIBANA_PRIVATE_KEY_PATH={'SET' if key_path else 'NOT SET'}, "
                    f"TACHIBANA_SECOND_PASSWORD={'SET' if password else 'NOT SET'}")
        sys.exit(1)

    if not os.path.exists(key_path):
        logger.error(f"Private key file not found at path: {key_path}")
        sys.exit(1)

    logger.info("Initializing TachibanaBrokerClient on DEMO environment...")
    config = BrokerConfig(
        provider="tachibana",
        api_url=api_url,
        api_token=auth_id,
        api_password=password,
        request_timeout=15,
        margin_trade_type=3, 
        account_type=4,      
        extra={"private_key_path": key_path}
    )

    try:
        client = create_broker(config)
    except Exception as e:
        logger.exception("Failed to instantiate broker client")
        sys.exit(1)

    try:
        logger.info("Running health check / login...")
        # Direct login call for verification
        client._client.login()
        logger.info("Login successful! Virtual URLs decrypted successfully.")

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
        
        logger.info("Login test passed successfully!")

    except Exception as e:
        logger.exception("An error occurred during demo login execution")
        sys.exit(2)
    finally:
        client.close()

if __name__ == "__main__":
    main()
