# Reproduces the Linux/CUDA userland from specification/environments.txt.
# The NVIDIA driver is supplied by the host and cannot be pinned in an image.
FROM --platform=linux/amd64 nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc-13=13.3.0-6ubuntu2~24.04.1 \
        g++-13=13.3.0-6ubuntu2~24.04.1 \
        python3=3.12.3-0ubuntu2 \
        python3-pip \
        python3-venv \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY environment/requirements-linux-gpu.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages -r /tmp/requirements.txt

COPY . /workspace
CMD ["python3", "environment/verify_environment.py", "--target"]

