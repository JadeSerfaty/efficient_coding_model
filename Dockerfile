# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    gcc \
    g++ \
    make \
    libhdf5-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Set HDF5_DIR environment variable
ENV HDF5_DIR=/usr/include/hdf5/serial

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /efficient_coding_model/

# Copy the application code to the container
COPY src /efficient_coding_model/src/
COPY run_model_runner.py /efficient_coding_model/
COPY requirements.txt .

# Install all dependencies except netCDF4 with increased timeout
RUN grep -v 'netCDF4' requirements.txt > temp_requirements.txt && pip install --default-timeout=100 --no-cache-dir -r temp_requirements.txt

# Install netCDF4 directly from GitHub
RUN pip install git+https://github.com/Unidata/netcdf4-python.git

# Define the entrypoint script to accept arguments
ENTRYPOINT ["python", "run_model_runner.py"]