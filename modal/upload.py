import os
import re
import numpy as np
import time
import shutil
import json
import matplotlib.pyplot as plt
from huggingface_hub import login, create_repo, upload_folder, HfFolder
from pathlib import Path # Using pathlib for easier path manipulation

# --- Configuration Constants ---
# Model and Repo Details
BASE_MODEL_NAME = "OpenGVLab/InternVideo2_distillation_models"
TARGET_REPO_NAME = "qingy2024/Window-IV2-CKPT" # Specify your target repo

# Training Parameters (Update if necessary)
TOTAL_STEPS = 24912 # Total expected steps for progress calculation

# Monitoring Configuration
CHECKPOINT_DIR_PATTERN = re.compile(r"^ckpt_iter(\d+)\.pth$")
POLL_INTERVAL_SECONDS = 30
PRE_UPLOAD_DELAY_SECONDS = 10 # Delay after finding checkpoint before processing

# --- Global State ---
# Set to track uploaded checkpoints (using Path objects for consistency)
uploaded_checkpoints = set()

# --- Helper Functions ---

def get_huggingface_token():
    """Retrieves the Hugging Face token from environment variable or login cache."""
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token:
        print("Using Hugging Face token from HUGGINGFACE_TOKEN environment variable.")
        return token
    token = HfFolder.get_token()
    if token:
        print("Using Hugging Face token from saved credentials.")
        return token
    raise ValueError("Hugging Face token not found. Set HUGGINGFACE_TOKEN environment variable or login using `huggingface-cli login`.")

# --- Core Logic ---

def find_new_checkpoint(current_dir: Path = Path('.')) -> tuple[int, Path] | None:
    """
    Finds the checkpoint folder in the specified directory with the highest
    step number that has not been previously uploaded.

    Args:
        current_dir (Path): The directory to scan for checkpoints.

    Returns:
        tuple[int, Path] | None: A tuple containing the (checkpoint_number, folder_path)
                                 or None if no new checkpoint is found.
    """
    new_checkpoints = []
    try:
        for item in current_dir.iterdir():
            if item.is_dir():
                match = CHECKPOINT_DIR_PATTERN.match(item.name)
                # Check if it matches the pattern AND has not been uploaded
                if match and item not in uploaded_checkpoints:
                    checkpoint_number = int(match.group(1))
                    new_checkpoints.append((checkpoint_number, item))
    except FileNotFoundError:
        print(f"Error: Directory not found: {current_dir}")
        return None
    except Exception as e:
        print(f"Error scanning directory {current_dir}: {e}")
        return None

    if new_checkpoints:
        new_checkpoints.sort(key=lambda x: x[0], reverse=True) # Sort by step number, highest first
        return new_checkpoints[0] # Return the one with the highest step number
    return None

def upload_checkpoint_to_hf(folder_path: Path, checkpoint_number: int, repo_id: str):
    """
    Uploads the checkpoint folder to Hugging Face Hub and deletes
    the folder locally upon successful upload.

    Args:
        folder_path (Path): Path to the local checkpoint folder.
        checkpoint_number (int): The checkpoint step number.
        repo_id (str): The Hugging Face repository ID (e.g., "username/repo-name").
    """
    print(f"\nAttempting to upload {folder_path.name} to Hugging Face repository: {repo_id}...")

    try:
        # Ensure repository exists
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"Repository {repo_id} exists or was created.")

        # Upload the folder contents
        upload_folder(
            folder_path=str(folder_path), # upload_folder expects string path
            repo_id=repo_id,
            commit_message=f"Upload checkpoint {checkpoint_number}",
            repo_type="model" # Explicitly set repo type
        )
        print(f"Successfully uploaded contents of {folder_path.name} to {repo_id}.")

        # Delete the local folder ONLY after successful upload
        try:
            shutil.rmtree(folder_path)
            print(f"Successfully deleted local folder: {folder_path}")
            return True # Indicate success
        except OSError as e:
            print(f"Error deleting local folder {folder_path}: {e}. Please delete manually.")
            return True # Upload succeeded, but deletion failed

    except Exception as e:
        print(f"ERROR during Hugging Face upload for {folder_path.name}: {e}")
        print("Upload failed. Local folder will not be deleted.")
        return False # Indicate failure

# --- Main Execution ---

def main():
    """
    Main loop to monitor for new checkpoints, upload them to Hugging Face Hub,
    and clean up locally.
    """
    try:
        hf_token = get_huggingface_token()
        login(hf_token)
        print("\nSuccessfully logged into Hugging Face Hub.")
    except ValueError as e:
        print(f"Error: {e}")
        return # Exit if login fails
    except Exception as e:
        print(f"An unexpected error occurred during Hugging Face login: {e}")
        return

    print("\nStarting checkpoint monitor...")
    print(f"Will check for new checkpoints matching '{CHECKPOINT_DIR_PATTERN.pattern}' every {POLL_INTERVAL_SECONDS} seconds.")
    print(f"Target repository: {TARGET_REPO_NAME}")
    print(f"Found checkpoints will be tracked (not re-uploaded): {uploaded_checkpoints or 'None yet'}")
    print("-" * 30)

    while True:
        new_checkpoint_info = find_new_checkpoint()

        if new_checkpoint_info:
            checkpoint_number, folder_path = new_checkpoint_info
            print(f"\nFound new checkpoint: {folder_path.name} (Step {checkpoint_number})")

            # Optional delay: wait a bit in case files are still being written
            print(f"Waiting {PRE_UPLOAD_DELAY_SECONDS} seconds before processing...")
            time.sleep(PRE_UPLOAD_DELAY_SECONDS)

            # Attempt upload and deletion
            upload_successful = upload_checkpoint_to_hf(
                folder_path=folder_path,
                checkpoint_number=checkpoint_number,
                repo_id=TARGET_REPO_NAME
            )

            if upload_successful:
                # Add to uploaded set ONLY if upload (and optionally deletion) was processed
                uploaded_checkpoints.add(folder_path)
                print(f"Added {folder_path.name} to the set of processed checkpoints.")

            print("-" * 30) # Separator after processing a checkpoint

        else:
            # Use \r for inline update when no checkpoint found
            print(f"\rNo new checkpoints found. Checking again in {POLL_INTERVAL_SECONDS} seconds... ", end="")

        # Wait before the next check
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    # Define necessary constants if they were outside the rewrite block but used here
    # Ensure BASE_MODEL_NAME and TARGET_REPO_NAME are defined globally or passed appropriately if needed.
    # Note: BASE_MODEL_NAME and TOTAL_STEPS are no longer used in this rewritten section.
    # TARGET_REPO_NAME is still used.
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
