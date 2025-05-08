# syntax=docker/dockerfile:1
# Use the latest Ubuntu base image
FROM condaforge/miniforge3:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install build-essential and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    ca-certificates \
    bzip2 \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    vim \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for the app
WORKDIR /app

# Copy the app into the image
COPY . /app

# Create a conda environment and install dependencies
RUN conda create -n dashapp python=3.11.10 && conda clean --all -y

# make the virtual env the one that starts by defualt after this line we should always be building in the virtual env not base
RUN echo "source activate dashapp" >> ~/.bashrc && echo "source activate dashapp" >> ~/.bash_profile
ENV PATH /opt/conda/envs/dashapp/bin:$PATH

# RUN --mount=type=secret,id=TOKENFILE \
#     git config --global url."https://username:$(cat /run/secrets/TOKENFILE)@github".insteadOf "https://github" && \
#     pip install --upgrade pip && \
#     pip install .  && \
#     pip cache purge
RUN pip install --upgrade pip && \
    pip install . && \
    pip cache purge

# Expose the port the app runs on for example 443
EXPOSE 80

# Run the app
CMD ["conda", "run", "--no-capture-output", "-n", "dashapp", "python", "/app/app/app.py", "--port", "80", "--host", "0.0.0.0", "--debug_off" ]
