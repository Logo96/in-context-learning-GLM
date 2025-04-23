from huggingface_hub import create_repo, upload_folder

repo_id = "joseph-tennyson/poisson-0.32" 
local_dir = "/root/in-context-learning-GLM/models/poisson-0.32"

upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")