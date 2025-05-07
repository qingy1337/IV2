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
    .pip_install(
        "huggingface_hub",
    )
    # speedier HF downloads
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_TOKEN": "lIpxcDdlfsJzTIAqZZAvGUKaYdlyOPrpLI_fh"[::-1]
    })
)

# ── Modal App / GPU entrypoint ───────────────────────────────────────────
app = modal.App(image=image, name="K600 Upload")

k600_volume = modal.Volume.from_name("k600", create_if_missing=True)

@app.function(volumes={"/root/k600": k600_volume}, timeout=86_400,cpu=4.0,memory=32768)
def upload_k600():
    import os
    import subprocess

    use_hf_large_folder = False

    if use_hf_large_folder:
        os.chdir("/root/k600/")

        subprocess.run(["huggingface-cli","upload-large-folder","qingy2024/Slim_K600","--repo-type=dataset",".","--num-workers=16"], check=True)
    else:
        os.chdir("/root/k600/kinetics-dataset/k600/train/train")
        subprocess.run("python upload_slim_kinetics_mp.py".split(), check=True)

    k600_volume.commit()

@app.local_entrypoint()
def main():
    upload_k600.remote()
