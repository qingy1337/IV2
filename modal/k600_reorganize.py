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
app = modal.App(image=image, name="K600 Reorganize")

k600_volume = modal.Volume.from_name("k600", create_if_missing=True)

@app.function(volumes={"/root/k600": k600_volume}, timeout=86_400)
def download_k600():
    import os
    import subprocess

    os.chdir("/root/k600/kinetics-dataset")

    subprocess.run(['git', 'pull'], check=True)

    if os.path.exists('./k600_reorganize.py'):
        subprocess.run(['mv', 'k600_reorganize.py', './k600/'], check = True)

    os.chdir("/root/k600/kinetics-dataset/k600")

    subprocess.run(['python', 'k600_reorganize.py', 'annotations/train.txt'], check=True)

    k600_volume.commit()

@app.local_entrypoint()
def main():
    download_k600.remote()
