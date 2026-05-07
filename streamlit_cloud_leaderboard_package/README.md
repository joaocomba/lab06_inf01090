# Streamlit Community Cloud Leaderboard (Google Sheets backend)

This package is designed for Streamlit Community Cloud.

## What it does

- Students upload a CSV submission
- The app scores it against a hidden ground truth stored in secrets
- The leaderboard is stored in a Google Sheet
- Students see rankings live

## Files

- `app.py` — Streamlit app
- `requirements.txt` — dependencies
- `.streamlit/secrets.example.toml` — example secrets file
- `README.md` — setup instructions

## Setup overview

1. Create a Google Sheet with a worksheet named `leaderboard`
2. Create a Google service account and share the Sheet with it
3. Put your service-account credentials and hidden ground truth in Streamlit secrets
4. Deploy `app.py` on Streamlit Community Cloud

## SETUP
To create a Google Cloud service account and download a JSON key for the Google Sheets API, follow these steps:
1. Create a Project: Log in to the Google Cloud Console. Click the project dropdown in the top header and select New Project. Enter a descriptive name and click Create.
2. Enable Google Sheets API: In the left-hand navigation menu, go to APIs & Services > Library. Search for "Google Sheets API," select it, and click Enable. It is often recommended to also enable the Google Drive API to ensure full functionality with sheet files.
3. Create a Service Account: Navigate to IAM & Admin > Service Accounts. Click + Create Service Account. Provide a name and ID, then click Create and Continue. You may optionally assign a role like "Editor" or "Owner," though for basic sheet access, you can skip this and click Done.
4. Create a JSON Key: In the Service Accounts list, click on the email of the account you just created. Navigate to the Keys tab at the top. Click Add Key > Create new key.
5. Download the Key File: Select JSON as the key type and click Create. The key file will automatically download to your computer. Store this JSON key securely and never share it publicly, as it provides direct access to your resources. 

## Hidden ground truth format

Must contain the columns matching your config, e.g.:

```csv
Id,SalePrice
1461,208500
1462,181500
1463,223500
```

## Student submission format

```csv
Id,prediction
1461,210000
1462,179500
1463,220000
```# labo5_INF01090
