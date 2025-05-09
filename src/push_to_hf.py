from huggingface_hub import create_repo, upload_folder, login


repo_name = "joint-1.2-curr-2"
local_name = "joint-1.2-curr-2"

repo_id = f"icl-182/{repo_name}" 
local_dir = f"/root/in-context-learning-GLM/models/{local_name}"

create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")