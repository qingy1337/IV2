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

    os.system('huggingface-cli download qingy2024/InternVideo2-Data InternVideo2_6B_V5_ACT75_eval.py --local-dir /root/ --repo-type dataset')

    command = "cd /root/ && python3 InternVideo2_6B_V5_ACT75_eval.py --num_frames 8"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    print("STDOUT:")
    print(stdout.decode())
    print("STDERR:")
    print(stderr.decode())

# Define a local entrypoint function for the Modal application.
@app.local_entrypoint()
def main():
    # When the script is run locally, this function is called.
    # It then calls the 'runwithgpu' function to be executed remotely on Modal.
    runwithgpu.remote()
