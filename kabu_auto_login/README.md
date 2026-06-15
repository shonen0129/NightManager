# kabu_auto_login_min

Minimal folder for running kabuSTATION auto-login inside a Windows VM.

## Setup (Windows VM)

1) Create a virtual environment and install dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) Place your Gmail API OAuth client file here:

- creds/credentials.json

3) Set environment variables (or create a .env file):

```powershell
$env:KABU_ACCOUNT_NUMBER="your_account_number"
$env:KABU_PASSWORD="your_password"
```

4) Run the script:

```powershell
python kabu_auto_login.py
```

## Notes

- The script expects kabuSTATION to be pinned as taskbar item 1 (Win + 1).
- Keep the VM screen active while the automation runs.
- The first Gmail run opens a browser for OAuth consent and stores creds/token.json.
