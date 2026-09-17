import os
from huggingface_hub import HfApi

from model_app.model_utility import LATEST_MODEL_PATH

TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("REPO_ID")
# REPO_ID = "your-username/model-name"
if not TOKEN:
    raise ValueError("HF_TOKEN not found")
if not REPO_ID:
    raise ValueError("REPO_ID not found")


LOCAL_FOLDER = LATEST_MODEL_PATH

api = HfApi()

api.create_repo(
    repo_id=REPO_ID,
    token=TOKEN,
    repo_type="model",
    exist_ok=True
)

print(f"Uploading {LOCAL_FOLDER} into {REPO_ID}...")

api.upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=REPO_ID,
    repo_type="model",
    token=TOKEN,
    commit_message="Initial model upload",
    ignore_patterns=["__pycache__", "*.log"]
)

print("Upload completed!")
