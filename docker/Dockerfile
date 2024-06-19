FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Install any additional dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt before installing dependencies
COPY requirements.txt /workspace/requirements.txt

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r /workspace/requirements.txt

# Set the working directory
WORKDIR /workspace s

# Expose the JupyterLab port
EXPOSE 8888