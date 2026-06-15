"""
kabuステーション自動ログイン機能

口座番号入力 → パスワード入力 → ワンタイムパスコード入力 のシンプルなフロー対応
参考: https://note.com/hraps/n/n5f9b2092a6a5

要件:
- PyAutoGUI & PyGetWindowでGUI自動操作
- Gmail APIでワンタイムコード取得
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

try:
    import pyautogui as pag
    import pygetwindow as pgw
    import pyperclip
except ImportError:
    raise ImportError(
        "Required packages not installed. Install with:\n"
        "pip install pyautogui pygetwindow pyperclip google-auth-oauthlib google-auth-httplib2 google-api-python-client"
    )


logger = logging.getLogger(__name__)
DEFAULT_BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path_str: str, base_dir: Path = DEFAULT_BASE_DIR) -> str:
    """Resolve a path relative to the script directory when needed."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


@dataclass
class LoginConfig:
    """ログイン設定"""

    account_number: str
    password: str
    gmail_credentials_path: str = "creds/credentials.json"
    wait_short: float = 2.0
    wait_long: float = 30.0
    otp_wait_timeout: int = 300  # 5分
    login_window_title: str = "ログイン"
    main_window_title: str = "kabuステーション"
    otp_sender_email: str = "no-reply@mail.kabu.com"
    otp_subject_keyword: str = "認証"
    passkey_window_title: str = "パスキー"
    passkey_skip_key: str = "tab"
    passkey_skip_presses: int = 7
    passkey_confirm_key: str = "enter"
    passkey_wait_timeout: float = 30.0
    account_number_wait: float = 30.0

    def __post_init__(self) -> None:
        self.gmail_credentials_path = resolve_path(self.gmail_credentials_path)


class GuiHelper:
    """PyAutoGUI/PyGetWindowのラッパークラス"""

    def __init__(self):
        """コンストラクタ"""
        self.logger = logging.getLogger(__class__.__name__)
        pag.FAILSAFE = False  # フェイルセーフ無効化（Ctrl+Alt+Deleteで中断可能）

    def sleep(self, seconds: float) -> None:
        """指定秒数だけ待機する"""
        pag.sleep(seconds)

    def get_window_by_title(self, title: str, timeout: float = 10.0):
        """
        指定タイトルに一致するウィンドウを取得

        Args:
            title: ウィンドウタイトルの一部
            timeout: タイムアウト時間（秒）

        Returns:
            ウィンドウオブジェクトまたはNone
        """
        interval = 1.0
        retry = int(timeout // interval)
        self.logger.debug(f"Looking for window: {title} (timeout={timeout}s)")

        for attempt in range(retry):
            windows = pgw.getWindowsWithTitle(title)
            if windows:
                self.logger.debug(f"Found window: {windows[0].title}")
                return windows[0]
            pag.sleep(interval)

        self.logger.warning(f"Window not found: {title}")
        return None

    def wait_for_window_to_disappear(self, title: str, timeout: float = 10.0) -> bool:
        """指定タイトルのウィンドウが消えるまで待機する"""
        interval = 1.0
        end_time = time.time() + timeout
        self.logger.debug(f"Waiting for window to disappear: {title}")

        while time.time() < end_time:
            if not pgw.getWindowsWithTitle(title):
                self.logger.debug(f"Window disappeared: {title}")
                return True
            pag.sleep(interval)

        self.logger.warning(f"Window still present after {timeout}s: {title}")
        return False

    def activate_window(self, window) -> None:
        """ウィンドウをアクティブにする"""
        if window:
            window.activate()
            pag.sleep(0.5)

    def write(
        self,
        text: str,
        interval: float = 0.05,
        force_clipboard: bool = False,
        clear_first: bool = False,
    ) -> None:
        """
        テキストを入力する
        日本語や特殊記号はクリップボード経由で入力

        Args:
            text: 入力テキスト
            interval: 文字間隔（秒）
            force_clipboard: クリップボード経由で入力する
            clear_first: 入力前に既存テキストをクリアする
        """
        if clear_first:
            pag.hotkey("ctrl", "a")
            pag.press("backspace")
            pag.sleep(0.05)

        use_clipboard = force_clipboard

        # 特殊記号チェック（USキーボード/JISキーボード互換性問題）
        if not use_clipboard and re.search(r"[@^:\-_]", text):
            use_clipboard = True
        # 日本語チェック
        elif not use_clipboard and re.search(
            r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", text
        ):
            use_clipboard = True

        if use_clipboard:
            pyperclip.copy(text)
            pag.hotkey("ctrl", "v")
            pyperclip.copy("")  # クリップボードをクリア
            pag.sleep(interval)
        else:
            pag.write(text, interval=interval)
            pag.sleep(interval)

    def press(self, key: str, presses: int = 1, interval: float = 0.1) -> None:
        """キーを入力する"""
        pag.press(key, presses=presses, interval=interval)

    def hotkey(self, *keys: str, interval: float = 0.1) -> None:
        """キーの組み合わせを入力する"""
        pag.hotkey(*keys, interval=interval)

    def type_slowly(self, text: str, interval: float = 0.1) -> None:
        """ゆっくりテキストを入力（検出回避用）"""
        self.write(text, interval=interval)


class GmailWatcher:
    """Gmail API経由でワンタイムコードを監視・取得するクラス"""

    def __init__(self, credentials_path: str = "creds/credentials.json"):
        """
        初期化

        Args:
            credentials_path: credentials.jsonのパス
        """
        self.logger = logging.getLogger(__class__.__name__)
        self.credentials_path = resolve_path(credentials_path)
        self.service = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Gmail APIの認証（トークン永続化対応）"""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError:
            self.logger.error(
                "Gmail API libraries not installed. "
                "Install: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
            )
            return

        SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
        creds_dir = Path(self.credentials_path).parent
        token_path = creds_dir / "token.json"

        if not Path(self.credentials_path).exists():
            self.logger.error(f"Credentials file not found: {self.credentials_path}")
            return

        try:
            creds = None

            # 保存済みトークンがあれば読み込む
            if token_path.exists():
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
                self.logger.debug("Loaded saved token from token.json")

            # トークンが無効 or 期限切れの場合
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    # リフレッシュトークンで更新
                    self.logger.info("Refreshing expired token...")
                    creds.refresh(Request())
                else:
                    # 初回認証: ブラウザで認証フロー実行
                    self.logger.info("No valid token found. Starting browser auth flow...")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                # トークンを保存
                token_path.write_text(creds.to_json())
                self.logger.info(f"Token saved to {token_path}")

            self.service = build("gmail", "v1", credentials=creds)
            self.logger.info("Gmail API authenticated successfully")
        except Exception as e:
            self.logger.error(f"Gmail API authentication failed: {e}")

    def extract_otp(self, email_body: str) -> Optional[str]:
        """
        メール本文からワンタイムコード（6桁数字）を抽出

        Args:
            email_body: メール本文

        Returns:
            抽出されたOTPまたはNone
        """
        # 6桁の数字を検索
        match = re.search(r"\b(\d{6})\b", email_body)
        if match:
            return match.group(1)

        # 他の形式も試す（ハイフン区切りなど）
        match = re.search(r"(\d{3})-(\d{3})", email_body)
        if match:
            return match.group(1) + match.group(2)

        return None

    def watch_for_email(
        self,
        sender_email: str,
        subject_keyword: str = "認証",
        duration_seconds: int = 300,
        check_interval: int = 10,
    ) -> Optional[str]:
        """
        指定時間メールを監視してOTPを取得

        Args:
            sender_email: メール送信元アドレス
            subject_keyword: 件名キーワード
            duration_seconds: 監視時間（秒）
            check_interval: チェック間隔（秒）

        Returns:
            抽出されたOTPまたはNone
        """
        if not self.service:
            self.logger.error("Gmail API not authenticated")
            return None

        start_time = time.time()
        self.logger.info(
            f"Watching for email from {sender_email} (timeout: {duration_seconds}s)"
        )

        try:
            while time.time() - start_time < duration_seconds:
                # 新着メール検索
                query = f"from:{sender_email} subject:{subject_keyword} is:unread"
                results = (
                    self.service.users()
                    .messages()
                    .list(userId="me", q=query, maxResults=5)
                    .execute()
                )

                messages = results.get("messages", [])
                if messages:
                    # 最新メールを取得
                    msg_id = messages[0]["id"]
                    message = (
                        self.service.users()
                        .messages()
                        .get(userId="me", id=msg_id, format="full")
                        .execute()
                    )

                    # メール本文を取得
                    headers = message["payload"]["headers"]
                    subject = next(
                        (h["value"] for h in headers if h["name"] == "Subject"), ""
                    )

                    # メール本文を抽出
                    body = ""
                    if "parts" in message["payload"]:
                        for part in message["payload"]["parts"]:
                            if part["mimeType"] == "text/plain":
                                data = part["body"].get("data", "")
                                if data:
                                    import base64

                                    body = base64.urlsafe_b64decode(data).decode()
                                    break
                    else:
                        data = message["payload"]["body"].get("data", "")
                        if data:
                            import base64

                            body = base64.urlsafe_b64decode(data).decode()

                    self.logger.debug(f"Email subject: {subject}")

                    # OTPを抽出
                    otp = self.extract_otp(body)
                    if otp:
                        self.logger.info(f"OTP extracted: {otp}")
                        return otp

                pag.sleep(check_interval)

        except Exception as e:
            self.logger.error(f"Error watching emails: {e}")

        self.logger.warning("Email watch timeout - no OTP found")
        return None


class KabusAutoLogin:
    """kabuステーション自動ログインメインクラス"""

    def __init__(self, config: LoginConfig):
        """
        初期化

        Args:
            config: ログイン設定
        """
        self.logger = logging.getLogger(__class__.__name__)
        self.config = config
        self.gui = GuiHelper()
        self.gmail_watcher = GmailWatcher(config.gmail_credentials_path)

    def launch_application(self) -> bool:
        """
        kabuステーションアプリケーションを起動

        Returns:
            成功時True
        """
        self.logger.info("Launching kabuSTATION...")

        try:
            # Windows環境での起動（タスクバーピン留めアプリ）
            # Windowsキー + 1 でタスクバー1番目のアプリを起動
            self.gui.hotkey("win", "1")
            self.gui.sleep(self.config.wait_short)

            # ログイン画面を待つ
            login_window = self.gui.get_window_by_title(
                self.config.login_window_title, timeout=self.config.wait_long
            )

            if login_window:
                self.gui.activate_window(login_window)
                self.logger.info("Login window found and activated")
                return True
            else:
                self.logger.error("Login window not found")
                return False

        except Exception as e:
            self.logger.error(f"Error launching application: {e}")
            return False

    def enter_account_number(self) -> bool:
        """
        口座番号を入力

        Returns:
            成功時True
        """
        self.logger.info("Entering account number...")

        try:
            if self.config.account_number_wait > 0:
                self.logger.info(
                    f"Waiting {self.config.account_number_wait:.1f}s before entering account number"
                )
                self.gui.sleep(self.config.account_number_wait)
            else:
                self.gui.sleep(1.0)
            self.gui.write(
                self.config.account_number,
                interval=0.05,
                force_clipboard=True,
                clear_first=True,
            )
            self.gui.sleep(0.5)
            self.gui.press("enter")
            self.gui.sleep(self.config.wait_short)
            self.logger.info("Account number entered successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error entering account number: {e}")
            return False

    def enter_password(self) -> bool:
        """
        パスワードを入力

        Returns:
            成功時True
        """
        self.logger.info("Entering password...")

        try:
            self.gui.sleep(1.0)
            self.gui.write(
                self.config.password,
                interval=0.05,
                force_clipboard=True,
                clear_first=True,
            )
            self.gui.sleep(0.5)
            self.gui.press("enter")
            self.gui.sleep(self.config.wait_short)
            self.logger.info("Password entered successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error entering password: {e}")
            return False

    def enter_otp(self, otp: str) -> bool:
        """
        ワンタイムパスコードを入力

        Args:
            otp: 6桁のワンタイムパスコード

        Returns:
            成功時True
        """
        self.logger.info("Entering OTP...")

        try:
            self.gui.sleep(1.0)
            self.gui.write(otp, interval=0.1, force_clipboard=True, clear_first=True)
            self.gui.sleep(0.5)
            self.gui.press("enter")
            self.gui.sleep(self.config.wait_short)
            self.logger.info("OTP entered successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error entering OTP: {e}")
            return False

    def get_otp_from_gmail(
        self, sender_email: Optional[str] = None, subject_keyword: Optional[str] = None
    ) -> Optional[str]:
        """
        Gmail APIを使用してワンタイムコードを取得

        Args:
            sender_email: kabuステーションからのメール送信元
            subject_keyword: メール件名キーワード

        Returns:
            抽出されたOTPまたはNone
        """
        resolved_sender = sender_email or self.config.otp_sender_email
        resolved_subject = subject_keyword or self.config.otp_subject_keyword
        return self.gmail_watcher.watch_for_email(
            sender_email=resolved_sender,
            subject_keyword=resolved_subject,
            duration_seconds=self.config.otp_wait_timeout,
        )

    def handle_passkey_prompt(self) -> bool:
        """パスキー選択画面が出た場合に「パスキーなしで実行」を選択する"""
        title = self.config.passkey_window_title.strip()
        if not title:
            return True

        self.logger.info("Checking for passkey prompt...")
        window = self.gui.get_window_by_title(title, timeout=self.config.passkey_wait_timeout)
        if not window:
            self.logger.info("No passkey prompt detected")
            return True

        self.gui.activate_window(window)

        if self.config.passkey_skip_presses > 0:
            self.gui.press(
                self.config.passkey_skip_key,
                presses=self.config.passkey_skip_presses,
                interval=0.2,
            )

        self.gui.press(self.config.passkey_confirm_key)
        self.gui.sleep(self.config.wait_short)
        self.logger.info("Passkey prompt handled")
        return True

    def verify_login_success(self) -> bool:
        """ログイン成功をウィンドウ状態で検証する"""
        self.logger.info("Verifying login result...")

        login_title = self.config.login_window_title.strip()
        if login_title:
            if not self.gui.wait_for_window_to_disappear(
                login_title, timeout=self.config.wait_long
            ):
                self.logger.error("Login window did not disappear")
                return False

        main_title = self.config.main_window_title.strip()
        if main_title:
            main_window = self.gui.get_window_by_title(
                main_title, timeout=self.config.wait_long
            )
            if not main_window:
                self.logger.error("Main window not found after login")
                return False
            self.gui.activate_window(main_window)

        self.logger.info("Login verified successfully")
        return True

    def login(self, use_gmail: bool = True) -> bool:
        """
        kabuステーションに自動ログイン

        Args:
            use_gmail: Gmail APIでOTP取得するかどうか

        Returns:
            ログイン成功時True
        """
        self.logger.info("=== kabuSTATION Auto Login Start ===")

        # Step 1: アプリケーション起動
        if not self.launch_application():
            self.logger.error("Failed to launch application")
            return False

        # Step 2: 口座番号入力
        if not self.enter_account_number():
            self.logger.error("Failed to enter account number")
            return False

        # Step 3: パスワード入力
        if not self.enter_password():
            self.logger.error("Failed to enter password")
            return False

        # Step 4: OTP取得と入力
        if use_gmail:
            self.logger.info("Retrieving OTP from Gmail...")
            otp = self.get_otp_from_gmail()

            if not otp:
                self.logger.error("Failed to retrieve OTP from Gmail")
                return False

            if not self.enter_otp(otp):
                self.logger.error("Failed to enter OTP")
                return False

            if not self.handle_passkey_prompt():
                self.logger.error("Failed to handle passkey prompt")
                return False

            if not self.verify_login_success():
                self.logger.error("Login verification failed")
                return False
        else:
            self.logger.warning("Gmail disabled - manual OTP entry required")

        self.logger.info("=== kabuSTATION Auto Login Completed Successfully ===")
        return True


def setup_logging(log_level: int = logging.INFO) -> None:
    """ログ設定"""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """メイン処理"""
    setup_logging()

    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).with_name(".env")
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.info(f"Loaded .env from {env_path}")
        else:
            load_dotenv(override=True)
    except ImportError:
        logger.warning(
            "python-dotenv is not installed. Environment variables from .env will not be loaded."
        )

    account_number = os.getenv("KABU_ACCOUNT_NUMBER", "").strip()
    password = os.getenv("KABU_PASSWORD", "")

    if not account_number or account_number == "your_account_number":
        logger.error("KABU_ACCOUNT_NUMBER is not set. Update .env or environment variables.")
        return False

    if not password or password in {"your_password", "your_passward"}:
        logger.error("KABU_PASSWORD is not set. Update .env or environment variables.")
        return False

    # 設定例
    config = LoginConfig(
        account_number=account_number,
        password=password,
        gmail_credentials_path="creds/credentials.json",
    )

    # ログイン実行
    auto_login = KabusAutoLogin(config)
    success = auto_login.login(use_gmail=True)

    if success:
        logger.info("✓ ログインに成功しました")
    else:
        logger.error("✗ ログインに失敗しました")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
