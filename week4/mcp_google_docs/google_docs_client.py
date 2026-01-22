import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

SCOPES = [
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.readonly"
]

def get_credentials():
    creds = None

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return creds


def search_insurance_docs(query: str):
    creds = get_credentials()

    drive_service = build("drive", "v3", credentials=creds)
    docs_service = build("docs", "v1", credentials=creds)

    # 1️⃣ Search files in Drive
    response = drive_service.files().list(
        q="mimeType='application/vnd.google-apps.document' "
          f"and fullText contains '{query}'",
        fields="files(id, name)"
    ).execute()

    results = []

    for file in response.get("files", []):
        doc = docs_service.documents().get(documentId=file["id"]).execute()

        text = []
        for element in doc.get("body").get("content"):
            if "paragraph" in element:
                for run in element["paragraph"].get("elements", []):
                    if "textRun" in run:
                        text.append(run["textRun"]["content"])

        results.append({
            "title": file["name"],
            "content": "".join(text)
        })

    return results
