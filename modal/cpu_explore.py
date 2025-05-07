import os
import random
import secrets
import subprocess
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .pip_install("jupyterlab")
    .pip_install("ipywidgets")
    .pip_install("hf_transfer")
    .pip_install('huggingface_hub')
    .pip_install("jupyter")
    .pip_install('packaging')
    .pip_install('ninja')
    .pip_install('torch')
    .pip_install('torchvision')
    .pip_install('numpy')
    .pip_install('wandb')
    .pip_install('datasets')
)

commands = """

apt-get update -y
apt-get install git curl -y

"""

commands = commands.strip().split('\n')

for command in commands:
    image = image.run_commands(command)

image = image.env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1"}  # turn on faster downloads from HF
)

k600_volume = modal.Volume.from_name("k600")

app = modal.App("CPU Ops", image=image)

k600_volume = modal.Volume.from_name("k600")

@app.function(timeout=86400, memory=20480, volumes={"/root/k600": k600_volume})
def run_jupyter():
    token = secrets.token_urlsafe(13)
    with modal.forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        print('-'*50 + '\n' + f"{url}\n"+'-'*50)
        subprocess.run(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                "--port=8888",
                "--LabApp.allow_origin='*'",
                "--LabApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
            stderr=subprocess.DEVNULL,
        )

@app.local_entrypoint()
def main(start_jupyter: bool = True):
    if start_jupyter:
        run_jupyter.remote()
    else:
        pass
