ARG AG_BENCH_BASE_IMAGE=nvidia/cuda:12.2.0-runtime-ubuntu22.04
FROM ${AG_BENCH_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV RUNNING_IN_DOCKER=true

RUN apt-get update && \
    apt-get install -y software-properties-common build-essential && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv ffmpeg libsm6 libxext6 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install utilities and AWS CLI
RUN apt-get install -y wget unzip curl git pciutils && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip -q awscliv2.zip && \
    ./aws/install && \
    rm awscliv2.zip && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /usr/local/aws

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Initialize conda and create mlzero environment
RUN conda init bash && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n mlzero python=3.11 -y && \
    conda create -n maab python=3.11 -y && \
    conda clean -a -y

# Make conda activate mlzero by default
RUN echo "conda activate mlzero" >> ~/.bashrc

# Set environment variables to use mlzero environment
ENV CONDA_DEFAULT_ENV=mlzero
ENV PATH="/opt/conda/envs/mlzero/bin:$PATH"

# Copy your setup_and_run script into the image
COPY setup_and_run.sh /setup_and_run.sh

# Make the setup_and_run script executable
RUN chmod +x /setup_and_run.sh

# (Optional) Copy your application code if needed
# COPY . /app
# WORKDIR /app

# (Optional) Install Python dependencies for MAAB in the mlzero environment
# COPY requirements.txt .
# RUN /opt/conda/envs/mlzero/bin/pip install --no-cache-dir -r requirements.txt

# Set the setup_and_run
ENTRYPOINT ["/setup_and_run.sh"]
