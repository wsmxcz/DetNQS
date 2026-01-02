# DetNQS: An Efficient Framework for Neural Network Quantum States in ab-initio Quantum Chemistry

[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![Python](https://img.shields.io/badge/Python-3.12%2B-EE4C2C.svg)](https://www.python.org/)
[![Build System](https://img.shields.io/badge/Build-CMake-green.svg)](https://cmake.org/)
[![JAX](https://img.shields.io/badge/JAX-0.8%2B-EE4C2C.svg)](https://docs.jax.dev/)


## Installation

### 1. Create Conda Environment
First, create the base environment using the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate detnqs
```

### 2. Install NVIDIA MathDX
The project requires **MathDX 25.12.0**. It must be installed manually into your conda environment directory to satisfy build dependencies.

```bash
# 1. Download the CUDA 12 compatible package
wget https://developer.nvidia.com/downloads/compute/cuSOLVERDx/redist/cuSOLVERDx/cuda12/nvidia-mathdx-25.12.0-cuda12.tar.gz

# 2. Extract
tar -xzf nvidia-mathdx-25.12.0-cuda12.tar.gz --strip-components=1 -C "$CONDA_PREFIX"

# 3. Verify the installation
ls "$CONDA_PREFIX/nvidia/mathdx/25.12/include/cusolverdx.hpp"
ls "$CONDA_PREFIX/nvidia/mathdx/25.12/include/cusolverdx_io.hpp"
```

### 3. Build and Install Package
With the environment and MathDX in place, install the package in editable mode.

```bash
pip install -e . -v --no-build-isolation \
  -Cbuild-dir=build \
  --config-settings=cmake.define.detnqs_CUDA_ARCHS=80 \
  --config-settings=cmake.define.detnqs_CUSOLVERDX_SM=800
```



## License

This project is licensed under the **Apache 2.0 License**. See the `LICENSE` file for more details.