import os
import time
import secrets
from random import randint
import subprocess
import pathlib
import modal

# ── Image ─────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim()
    # core Jupyter stack
    .pip_install(
        "httpx",
        "tqdm",
        "hf_transfer",
        "packaging",
        "ninja",
    )
    # OS‑level bits
    .run_commands(
        "apt-get update -y"
    )
    .apt_install(
        "git",
        "curl",
        "wget",
        "ffmpeg",
        "aria2"
    )
)

# ── Modal App / GPU entrypoint ───────────────────────────────────────────
app = modal.App(image=image, name="K600 Clean")

k600_volume = modal.Volume.from_name("k600", create_if_missing=True)

@app.function(volumes={"/root/k600": k600_volume}, timeout=86_400)
def download_k600():
    import os
    import subprocess

    # Change directory to the dataset root
    os.chdir("/root/k600/kinetics-dataset")

    # Pull the latest changes from the repo
    subprocess.run(['git', 'pull'], check=True)

    # Move removed_actions.txt to the k600 subdirectory
    subprocess.run(['mv', 'removed_actions.txt', './k600/'], check=True)

    # Change to the k600 directory
    os.chdir("/root/k600/kinetics-dataset/k600")

    # Run the grep | tr | xargs command using shell=True
    subprocess.run(
        'grep -v "^$" removed_actions.txt | tr "\n" "\0" | xargs -0 -I {} rm -rf "./train/train/{}/"',
        shell=True,
        check=True
    )

    # Commit the volume changes (if needed by your framework)
    k600_volume.commit()

@app.local_entrypoint()
def main():
    download_k600.remote()
