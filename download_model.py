import argparse
import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
from config import AVAILABLE_MODELS, LOCAL_MODELS_DIR

load_dotenv()  # Needed for HuggingFace token


def download_model(repo_id: str, local_name: str = None):
    # Default to the repository name if no local name is provided
    if local_name is None:
        local_name = repo_id.split("/")[-1]
    
    local_dir = os.path.join(".", LOCAL_MODELS_DIR, local_name)
    
    print(f"Fetching '{repo_id}' from Hugging Face...")
    print(f"Target directory: {local_dir}")

    # Create the target directory (and any missing parents) if it doesn't exist yet
    os.makedirs(local_dir, exist_ok=True)

    # Download the entire repository
    # We set local_dir_use_symlinks=False to actually copy the files into the folder
    # instead of just symlinking to the huggingface cache
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False, 
        resume_download=True
    )
    
    print(f"✅ Successfully downloaded '{repo_id}' to {local_dir}")


if __name__ == "__main__":
    print("You can download one of the following models that are currently supported:")
    for i, model_repo_name in enumerate(AVAILABLE_MODELS):
        print(f"{i} - {model_repo_name}")
    
    print("\nPlease type only the single model number NOT the name.")
    valid_values = tuple(range(1, len(AVAILABLE_MODELS) + 1))
    model_repo_index = int(input())
    while model_repo_index not in valid_values:
        print("The model number you selected is invalid. Please try again.")
        model_repo_index = int(input())
    
    selected_model_repo = AVAILABLE_MODELS[model_repo_index]
    model_name = selected_model_repo.split("/")[-1]
    print(
        f"You selected {selected_model_repo} and it will now be downloaded "
        "in the {LOCAL_MODELS_DIR} directory"
    )
    download_model(selected_model_repo, model_name)
