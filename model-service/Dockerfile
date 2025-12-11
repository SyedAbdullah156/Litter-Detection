FROM continuumio/miniconda3:latest

WORKDIR /app

# Install conda packages (PyTorch with CUDA support)
RUN conda install -y -c pytorch -c nvidia -c conda-forge \
    python=3.10.19 \
    pytorch=2.5.1 \
    torchvision=0.20.1 \
    pytorch-cuda=12.4 \
    numpy=2.2.6 \
    pillow=12.0.0 \
    && conda clean -afy

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install pip packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .

# Create necessary directories
RUN mkdir -p models temp

# Expose port
EXPOSE 8001

# Run the application
CMD ["python", "app.py"]