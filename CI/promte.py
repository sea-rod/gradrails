from huggingface_hub import snapshot_download, upload_folder
import os

REPO = os.environ.get("MODEL_PATH", "gaurdrails-miniLM-L6-V2")

path = snapshot_download(REPO, revision="dev")
upload_folder(
    repo_id=REPO,
    folder_path=path,
    revision="main",
    delete_patterns="*",
    commit_message="Promote dev to main",
    token=os.environ.get("HF_TOKEN"),
)
