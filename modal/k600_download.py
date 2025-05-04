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
app = modal.App(image=image, name="K600 Downloader")

k600_volume = modal.Volume.from_name("k600", create_if_missing=True)

@app.function(volumes={"/root/k600": k600_volume}, timeout=86_400)
def download_k600():
    import os
    import subprocess

    # Change to the mounted directory
    os.chdir("/root/k600")

    import shutil

    if not os.path.exists('kinetics-dataset'):
        subprocess.run(["git", "clone", "https://github.com/qingy1337/kinetics-dataset.git"], check=True)

    os.chdir("/root/k600/kinetics-dataset")

    subprocess.run(['git', 'pull'], check=True)

    subprocess.run(["bash", "./k600_downloader.sh"], check=True)
    subprocess.run(["bash", "./k600_extractor.sh"], check=True)

    k600_volume.commit()

@app.local_entrypoint()
def main():
    download_k600.remote()
