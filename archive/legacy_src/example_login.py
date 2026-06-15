#!/usr/bin/env python
"""
kabuステーション自動ログイン - 使用例

このスクリプトは kabu_auto_login.py の使用方法を示します。
"""

import sys
import logging
import os
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from kabu_auto_login import KabusAutoLogin, LoginConfig, setup_logging


def example_basic():
    """基本的な使用例"""
    print("=" * 60)
    print("例1: 基本的な自動ログイン（メール + パスワード + Gmail OTP）")
    print("=" * 60)

    # ログ設定
    setup_logging(log_level=logging.INFO)

    # ログイン設定（環境変数または直接指定）
    email = os.getenv("KABU_EMAIL") or "your_email@example.com"
    password = os.getenv("KABU_PASSWORD") or "your_password"

    config = LoginConfig(
        email=email,
        password=password,
        gmail_credentials_path="creds/credentials.json",
        wait_short=2.0,
        wait_long=30.0,
        otp_wait_timeout=300,  # 5分
    )

    # ログイン実行
    auto_login = KabusAutoLogin(config)
    success = auto_login.login(use_gmail=True)

    if success:
        print("\n✓ ログイン成功！kabuステーション API が使用可能です。")
    else:
        print("\n✗ ログイン失敗。エラーログを確認してください。")

    return success


def example_manual_otp():
    """手動OTP入力の例"""
    print("=" * 60)
    print("例2: 手動OTP入力（Gmail APIなし）")
    print("=" * 60)

    setup_logging(log_level=logging.INFO)

    config = LoginConfig(
        email="your_email@example.com",
        password="your_password",
    )

    auto_login = KabusAutoLogin(config)

    # ステップバイステップ実行
    print("\n[ステップ1] アプリケーション起動...")
    if not auto_login.launch_application():
        print("✗ 起動失敗")
        return False

    print("✓ 起動完了\n")
    print("[ステップ2] メールアドレス入力...")
    if not auto_login.enter_email():
        print("✗ 入力失敗")
        return False

    print("✓ 入力完了\n")
    print("[ステップ3] パスワード入力...")
    if not auto_login.enter_password():
        print("✗ 入力失敗")
        return False

    print("✓ 入力完了\n")

    # ここで手動でOTPをスマホから取得
    print("[ステップ4] スマホのメールからOTPを確認してください")
    otp = input("OTP（6桁）を入力: ").strip()

    if len(otp) == 6 and otp.isdigit():
        if auto_login.enter_otp(otp):
            print("✓ ログイン完了")
            return True

    print("✗ 無効なOTP")
    return False


def example_debug_mode():
    """デバッグモード"""
    print("=" * 60)
    print("例3: デバッグモード（詳細ログ出力）")
    print("=" * 60)

    # デバッグレベルでログ出力
    setup_logging(log_level=logging.DEBUG)

    config = LoginConfig(
        email="your_email@example.com",
        password="your_password",
        wait_long=60.0,  # デバッグ時は長めに
    )

    auto_login = KabusAutoLogin(config)

    # 各ステップで詳細ログが表示されます
    success = auto_login.login(use_gmail=True)

    return success


def example_gui_helper():
    """GuiHelper の低レベル操作例"""
    print("=" * 60)
    print("例4: GuiHelper での直接操作")
    print("=" * 60)

    setup_logging(log_level=logging.INFO)

    from kabu_auto_login import GuiHelper

    gui = GuiHelper()

    print("ログイン画面を検出中...")
    window = gui.get_window_by_title("ログイン", timeout=30)

    if window:
        print(f"✓ ウィンドウ検出: {window.title}")
        gui.activate_window(window)
        gui.sleep(1)

        # テキスト入力
        print("メールアドレスを入力...")
        gui.write("test@example.com", interval=0.05)
        gui.sleep(0.5)
        gui.press("enter")

        return True
    else:
        print("✗ ログイン画面が見つかりません")
        return False


def example_gmail_watch():
    """Gmail監視の単独実行例"""
    print("=" * 60)
    print("例5: Gmail OTP 取得テスト")
    print("=" * 60)

    setup_logging(log_level=logging.INFO)

    from kabu_auto_login import GmailWatcher

    watcher = GmailWatcher(credentials_path="creds/credentials.json")

    if not watcher.service:
        print("✗ Gmail API 認証失敗")
        return False

    print("5分間メールを監視中...")
    print("（kabuステーションのOTPメール送信後にEnter を押してください）")

    otp = watcher.watch_for_email(
        sender_email="noreply@kabu.com",
        subject_keyword="認証",
        duration_seconds=300,
        check_interval=10,
    )

    if otp:
        print(f"✓ OTP 取得成功: {otp}")
        return True
    else:
        print("✗ OTP 取得失敗")
        return False


def example_multiple_accounts():
    """複数アカウントのログイン例"""
    print("=" * 60)
    print("例6: 複数アカウント順次ログイン")
    print("=" * 60)

    setup_logging(log_level=logging.INFO)

    accounts = [
        {
            "email": "account1@example.com",
            "password": "password1",
            "name": "アカウント1",
        },
        {
            "email": "account2@example.com",
            "password": "password2",
            "name": "アカウント2",
        },
    ]

    results = {}

    for account in accounts:
        print(f"\n--- {account['name']} ログイン開始 ---")

        config = LoginConfig(
            email=account["email"],
            password=account["password"],
        )

        auto_login = KabusAutoLogin(config)
        success = auto_login.login(use_gmail=True)

        results[account["name"]] = success

        # アカウント切り替え待機
        if success:
            print(f"✓ {account['name']} ログイン完了")
            import time

            time.sleep(5)  # 次の操作まで5秒待機

    # 結果サマリー
    print("\n" + "=" * 60)
    print("ログイン結果サマリー")
    print("=" * 60)
    for name, success in results.items():
        status = "✓ 成功" if success else "✗ 失敗"
        print(f"{name}: {status}")


def main():
    """メイン処理"""
    print("\n" + "=" * 60)
    print("kabuステーション自動ログイン - 使用例集")
    print("=" * 60)
    print(
        """
使用方法:
  python example_login.py [例番号]

例:
  python example_login.py 1   # 基本的な使用例
  python example_login.py 2   # 手動OTP入力
  python example_login.py 3   # デバッグモード
  python example_login.py 4   # GuiHelper 直接操作
  python example_login.py 5   # Gmail OTP取得テスト
  python example_login.py 6   # 複数アカウント
"""
    )

    # コマンドライン引数を確認
    if len(sys.argv) > 1:
        example_num = sys.argv[1]

        examples = {
            "1": example_basic,
            "2": example_manual_otp,
            "3": example_debug_mode,
            "4": example_gui_helper,
            "5": example_gmail_watch,
            "6": example_multiple_accounts,
        }

        if example_num in examples:
            try:
                result = examples[example_num]()
                exit(0 if result else 1)
            except Exception as e:
                print(f"\n✗ エラーが発生しました: {e}")
                import traceback

                traceback.print_exc()
                exit(1)
        else:
            print(f"✗ 無効な例番号: {example_num}")
            print(f"   利用可能: {', '.join(examples.keys())}")
            exit(1)
    else:
        print("\n例番号を指定してください。例: python example_login.py 1")
        print("\n最初のログインテストをお勧めします:")
        print("  python example_login.py 1")
        exit(1)


if __name__ == "__main__":
    main()
