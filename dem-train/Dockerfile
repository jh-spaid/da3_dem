FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY trainer /workspace/trainer
COPY Depth-Anything-3 /workspace/Depth-Anything-3

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace:/workspace/Depth-Anything-3/src
#ENTRYPOINT ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "trainer.unet_ddp_wandb"]
#ENTRYPOINT ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "trainer.train_dem_v3_orig"]
ENTRYPOINT ["torchrun", "--standalone", "--nproc_per_node=1", "-m", "trainer.unet_wandb"]