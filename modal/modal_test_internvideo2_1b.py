import os
import secrets
from random import randint
import subprocess
import pathlib
import modal

image = (
    # Start by using a pre-built Docker image from NVIDIA GPU Cloud (NGC) registry.
    # This image comes with CUDA 12.6.3 and Ubuntu 22.04, optimized for GPU workloads.
    # We add Python 3.10 to this base image.
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.10")
    # Install JupyterLab, an interactive web-based development environment.
    .pip_install("jupyterlab")
    # Install ipywidgets, interactive HTML widgets for Jupyter notebooks.
    .pip_install("ipywidgets")
    # Install hf_transfer, a tool for faster downloads from Hugging Face Hub.
    .pip_install("hf_transfer")
    # Install jupyter core package.
    .pip_install("jupyter")
    # Install packaging utilities.
    .pip_install('packaging')
    # Install ninja build system.
    .pip_install('ninja')
    .run_commands(
        "apt-get update -y", # Update package lists for upgrades and new package installations.
        "apt-get install git -y", # Install git, a distributed version control system.
        "apt-get install curl -y" # Install curl, a command-line tool for transferring data with URLs.
    )
    # Install PyTorch, a deep learning framework. Version is pinned to be at least 2.4.0.
    .pip_install('torch')  # Updated torch installation with version constraint
    # Install torchvision, providing datasets and image transformations for PyTorch.
    .pip_install('torchvision')
    # Install numpy, a fundamental package for scientific computing with Python.
    .pip_install('numpy')
    # Install wandb (Weights & Biases), a platform for tracking and visualizing machine learning experiments.
    .pip_install('wandb')
    # Install pandas, a data manipulation and analysis library.
    .pip_install('pandas')
    # Install tensorboard, a visualization toolkit for machine learning experiments.
    .pip_install('tensorboard')
    # Install flash-attn, for faster attention mechanisms in transformers.
    .run_commands('pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.7.4.post1+cu126torch2.7-cp310-cp310-linux_x86_64.whl')
    .pip_install('flash-attn')
    # Install vllm, a library for fast and efficient large language model serving.
    # .pip_install('vllm')
    # Install pyzmq, Python bindings for ZeroMQ, a high-performance asynchronous messaging library.
    # .run_commands(
    #     "pip install -U pyzmq"
    # )
    # # Install Hugging Face libraries for natural language processing tasks.
    # .pip_install(
    #     "datasets==3.3.2", # Datasets library for accessing and sharing datasets.
    #     "accelerate==1.4.0", # Accelerate library for easy distributed training and inference.
    #     "evaluate==0.4.3", # Evaluate library for evaluating machine learning models.
    #     "bitsandbytes==0.45.3", # bitsandbytes for mixed-precision and quantization techniques.
    #     "trl==0.12.0", # Transformer Reinforcement Learning library.
    #     "peft==0.14.0", # Parameter-Efficient Fine-Tuning library.
    #     "protobuf", # Protocol Buffers - serialization library.
    #     "sentencepiece" # SentencePiece tokenizer for subword tokenization.
    # )
    # # Install unsloth dependencies, which may be related to specific optimization or fine-tuning techniques.
    # .pip_install('accelerate')
    # # .run_commands('pip install xformers==0.0.29.post3')  # Corrected xformers version - install specific version of xformers library for optimized transformer operations.
    # .pip_install('peft')
    # .pip_install('triton') # Triton, a language and compiler for writing highly efficient GPU programs.
    # # .pip_install('cut_cross_entropy') # Likely a custom or specific loss function.
    # # .pip_install('unsloth_zoo') # Potentially a collection of models or utilities from the Unsloth project.
    # .pip_install('sentencepiece')
    # .pip_install('datasets')
    .pip_install('huggingface_hub') # Hugging Face Hub library to interact with models, datasets, and spaces on the Hub.
    # # .pip_install('unsloth') # Unsloth library, potentially related to model optimization or training.
    # .pip_install('evaluate')
    # .pip_install('regex') # Regex library for regular expressions.
    # .pip_install('matplotlib') # Matplotlib, a plotting library for Python.
    # .pip_install(
    #     "einops",
    #     "timm",
    #     "iv2_utils",
    #     "open-clip-torch",
    #     "scipy",
    # )
)

# Set environment variables within the Modal image.
image = image.env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_TOKEN": "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",
    "WANDB_API_KEY": "0aee5395a94fbd8e33ada07b71309b1f30561cac"
})

# Add system-level commands to the Modal image to update package lists and install essential tools.
image = image.run_commands(
    "apt-get update -y", # Update package lists for upgrades and new package installations.
    "apt-get install git -y", # Install git, a distributed version control system.
    "apt-get install curl -y" # Install curl, a command-line tool for transferring data with URLs.
)

# Download pre-trained Qwen models from Hugging Face Hub and store them locally within the image at /root directory.
image = image.run_commands(
    # "huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /root/Qwen-2.5-3B-Instruct", # Download Qwen2.5-3B-Instruct model.
    "huggingface-cli download OpenGVLab/InternVideo2-Stage2_1B-224p-f4 --local-dir /root/",
    # "huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir /root/Qwen-2.5-14B-Instruct", # Download Qwen2.5-14B-Instruct model.
    # "huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir /root/Qwen-2.5-32B-Instruct" # Download Qwen2.5-32B-Instruct model.
)

# Clone the InternVideo2 Repository
image = image.run_commands(
    "cd /root/ && git clone https://github.com/qingy1337/IV2.git",
)

image = image.run_commands(
    "pip install opencv-python tabulate",
    "apt-get update",
    "apt-get install ffmpeg libsm6 libxext6 -y"
)
# ---------

# Create a Modal application named "Gradience".
app = modal.App(image=image, name="InternVideo2 Experiments")

# Define a Modal function named 'runwithgpu' that will be executed on Modal cloud.
@app.function(gpu="A100-80GB:1", timeout=86400) # Use an A100-80GB GPU and set a max execution time of 24 hours (86400 seconds).
def runwithgpu():
    import os
    import subprocess

    os.chdir("/root/IV2")

    commands = """
    git pull
    git checkout fix-iv2-1b
    pip install -r reqs.txt
    """.strip().splitlines()

    for line in commands:
        subprocess.run(line.strip().split(), check = True)

    pass

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
