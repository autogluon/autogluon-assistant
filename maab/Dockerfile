# Use an official Python image as a base (change as needed)
FROM python:3.11-slim

# Install any OS dependencies you need
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    which \
    && rm -rf /var/lib/apt/lists/*

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

# (Optional) Install Python dependencies in the mlzero environment
# COPY requirements.txt .
# RUN /opt/conda/envs/mlzero/bin/pip install --no-cache-dir -r requirements.txt

# Set the setup_and_run
ENTRYPOINT ["/setup_and_run.sh"]