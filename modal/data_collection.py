import os
import secrets
from random import randint
import subprocess
import pathlib
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.10")
    .pip_install("jupyterlab")
    .pip_install("ipywidgets")
    .pip_install("hf_transfer")
    .pip_install("jupyter")
    .pip_install('packaging')
    .pip_install('ninja')
    .pip_install('torch')
    .pip_install('torchvision')
    .pip_install('numpy')
    .pip_install('wandb')
    .pip_install('pandas')
    .pip_install('tensorboard')
    .run_commands('pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.7.4.post1+cu126torch2.7-cp310-cp310-linux_x86_64.whl')
    .pip_install('flash-attn')
    .pip_install('huggingface_hub')
)

image = image.env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_TOKEN": "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",
    "WANDB_API_KEY": "0aee5395a94fbd8e33ada07b71309b1f30561cac"
})

image = image.run_commands(
    "apt-get update -y",
    "apt-get install git -y",
    "apt-get install curl -y"
)

# Download model
image = image.run_commands(
    "huggingface-cli download OpenGVLab/InternVideo2-Stage2_1B-224p-f4 --local-dir /root/",
)

# # Clone the InternVideo2 Repository
# image = image.run_commands(
#     "cd /root/ && git clone https://github.com/qingy1337/IV2.git",
# )

image = image.run_commands(
    "pip install opencv-python tabulate",
    "apt-get update",
    "apt-get install ffmpeg libsm6 libxext6 -y"
)

# Install requirements from IV2 repo.
image = image.run_commands(
    "curl -s -o reqs.txt https://raw.githubusercontent.com/qingy1337/IV2/refs/heads/main/reqs.txt && pip install -r reqs.txt",
    "rm reqs.txt"
)
# ---------

# Create a Modal application.
app = modal.App(image=image, name="InternVideo2 Experiments")

# Define a Modal function named 'runwithgpu' that will be executed on Modal cloud.
@app.function(gpu="A100-40GB:1", timeout=86400)
def runwithgpu():
    import os
    import subprocess

    os.chdir("/root/")

    # Generate a secure random token for JupyterLab access authentication.
    token = 'tesslate'
    # Set up port forwarding to access JupyterLab running inside the Modal container from the local machine.
    with modal.forward(8888) as tunnel:
        # Construct the URL to access JupyterLab, including the generated token.
        url = tunnel.url + "/?token=" + token
        print('-'*50 + '\n' + f"{url}\n"+'-'*50) # Print the URL to the console, making it easy to access JupyterLab in a browser.
        # Start JupyterLab server with specific configurations.
        subprocess.run(
            [
                "jupyter", # Command to execute JupyterLab.
                "lab", # Start JupyterLab interface.
                "--no-browser", # Prevent JupyterLab from trying to open a browser automatically.
                "--allow-root", # Allow JupyterLab to be run as root user inside the container.
                "--ip=0.0.0.0", # Bind JupyterLab to all network interfaces, making it accessible externally.
                "--port=8888", # Specify the port for JupyterLab to listen on.
                "--LabApp.allow_origin='*'", # Allow requests from any origin (for easier access from different networks).
                "--LabApp.allow_remote_access=1", # Allow remote connections to JupyterLab.
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"}, # Set environment variables, including the authentication token and shell.
            stderr=subprocess.DEVNULL, # Suppress standard error output from JupyterLab for cleaner logs.
        )

# Define a local entrypoint function for the Modal application.
@app.local_entrypoint()
def main():
    # When the script is run locally, this function is called.
    # It then calls the 'runwithgpu' function to be executed remotely on Modal.
    runwithgpu.remote()
