"""leadlag/reporting/gmail_sender.py — Gmail API email sender with timeout.

This module is intentionally separate from ``kabu_auto_login.GmailWatcher``
because it uses the ``gmail.send`` OAuth scope (not ``gmail.readonly``).
A separate token file avoids overwriting the OTP-reader token.
"""

from __future__ import annotations

import base64
import concurrent.futures
import logging
import time
from collections.abc import Sequence
from email.message import EmailMessage
from pathlib import Path

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    _GMAIL_LIBS_AVAILABLE = True
except ImportError:
    Request = None  # type: ignore[assignment, misc]
    Credentials = None  # type: ignore[assignment, misc]
    InstalledAppFlow = None  # type: ignore[assignment, misc]
    build = None  # type: ignore[assignment]
    _GMAIL_LIBS_AVAILABLE = False

logger = logging.getLogger(__name__)

SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"
DEFAULT_SCOPES = [SEND_SCOPE]
DEFAULT_SEND_TIMEOUT = 30.0


def _check_gmail_libs() -> None:
    if not _GMAIL_LIBS_AVAILABLE:
        raise RuntimeError(
            "Gmail API libraries are not installed. "
            "Install the optional gmail extras: pip install -e '.[gmail]'"
        )


class GmailSender:
    """Send plain-text or HTML emails via the Gmail API.

    The constructor is safe for unattended automation: it will **not** run the
    interactive browser OAuth flow. If a valid ``token_path`` exists it is used;
    otherwise the first ``send()`` call raises a clear error. Use
    ``authorize_gmail_send()`` (or ``--authorize`` on the CLI) to create the
    initial token interactively.
    """

    def __init__(
        self,
        credentials_path: str | Path,
        token_path: str | Path,
        from_email: str | None = None,
        send_timeout: float = DEFAULT_SEND_TIMEOUT,
    ) -> None:
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.from_email = from_email
        self.send_timeout = send_timeout
        self._service = None

    def authenticate(self, *, allow_interactive: bool = False) -> None:
        """Authenticate with Gmail and build the API service.

        Args:
            allow_interactive: If True, open a browser for the initial OAuth
                consent flow. Must be ``False`` in automated close jobs.
        """
        _check_gmail_libs()
        if not self.credentials_path.exists():
            raise FileNotFoundError(
                f"Gmail credentials not found: {self.credentials_path}. "
                "Download the OAuth client credentials from Google Cloud Console."
            )

        creds: Credentials | None = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), DEFAULT_SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired Gmail send token...")
                creds.refresh(Request())
            else:
                if not allow_interactive:
                    raise RuntimeError(
                        f"No valid Gmail send token at {self.token_path}. "
                        "Run 'python tools/production/send_daily_close_pnl_report.py --authorize' once."
                    )
                logger.info("Starting Gmail send OAuth flow...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), DEFAULT_SCOPES
                )
                creds = flow.run_local_server(port=0)

            self.token_path.write_text(creds.to_json(), encoding="utf-8")
            logger.info("Gmail send token saved: %s", self.token_path)

        self._service = build(
            "gmail",
            "v1",
            credentials=creds,
            cache_discovery=False,
        )
        logger.info("Gmail API (send) authenticated")

    def create_message(
        self,
        to: Sequence[str],
        subject: str,
        body: str,
        from_email: str | None = None,
    ) -> dict:
        """Create a base64url-encoded MIME message for the Gmail API."""
        sender = from_email or self.from_email or "me"
        message = EmailMessage()
        message.set_content(body)
        message["To"] = ", ".join(to)
        message["From"] = sender
        message["Subject"] = subject
        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        return {"raw": encoded}

    def send(
        self,
        to: Sequence[str],
        subject: str,
        body: str,
        from_email: str | None = None,
        dry_run: bool = False,
    ) -> str | None:
        """Send a message and return the Gmail message ID.

        Args:
            to: List of recipient addresses.
            subject: Email subject.
            body: Plain-text body.
            from_email: Optional sender display address.
            dry_run: If True, only log; do not call the API.

        Returns:
            The Gmail message ID, or None in dry-run mode.
        """
        if dry_run:
            logger.info("[DRY RUN] Would send email to %s: %s", to, subject)
            return None

        if self._service is None:
            self.authenticate(allow_interactive=False)

        message = self.create_message(to, subject, body, from_email)

        # Wrap the API call in a timeout to prevent hangs during close jobs.
        # This follows the hang-prevention convention for broker/API calls.
        deadline = time.time() + self.send_timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._service.users().messages().send(userId="me", body=message).execute
            )
            try:
                result = future.result(timeout=self.send_timeout)
            except concurrent.futures.TimeoutError as e:
                raise TimeoutError(
                    f"Gmail API send timed out after {self.send_timeout}s"
                ) from e

        if deadline - time.time() < 0:
            logger.warning("Gmail send completed but exceeded the deadline; result may be stale")

        message_id = result.get("id")
        logger.info("Email sent via Gmail API. Message ID: %s", message_id)
        return message_id


def authorize_gmail_send(
    credentials_path: str | Path,
    token_path: str | Path,
    from_email: str | None = None,
) -> None:
    """Run the interactive Gmail send authorization flow once."""
    _check_gmail_libs()
    sender = GmailSender(credentials_path, token_path, from_email)
    sender.authenticate(allow_interactive=True)
    logger.info("Gmail send authorization complete. Token: %s", token_path)
