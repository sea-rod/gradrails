from huggingface_hub import HfApi
api = HfApi()
api.create_branch(repo_id="sea-rod/gaurdrails-miniLM-L6-V2", branch="dev", exist_ok=True)
api.upload_folder(
     folder_path="./output/gaurdrails-miniLM-L6-V2",
     repo_id="sea-rod/gaurdrails-miniLM-L6-V2",
     revision="dev",
     commit_message="update model",
 )