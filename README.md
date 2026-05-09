# CUDA-Efficient Hyperspectral Unmixing using MBUNet-TCN

This repository contains an MBUNet-inspired hyperspectral unmixing model implemented in PyTorch.  
The original MBUNet paper uses CNN + bidirectional Mamba modules for spatial-spectral feature extraction. Since `mamba-ssm` and `causal-conv1d` caused installation issues on free Kaggle/Colab GPU environments, this project replaces the Mamba sequence blocks with a CUDA-safe bidirectional Temporal Convolutional Network (TCN).

The goal is to perform hyperspectral unmixing on the AVIRIS Cuprite dataset under limited GPU resources.

## Project Summary

Hyperspectral unmixing decomposes each pixel spectrum into:

- **Endmembers**: pure material/mineral spectra
- **Abundances**: fractional contribution of each endmember per pixel

This project implements:

- Dual-stream spatial-spectral architecture inspired by MBUNet
- CNN-based local spatial and spectral feature extraction
- Bidirectional TCN sequence blocks as a lightweight Mamba replacement
- Shared decoder weights representing learned endmembers
- VCA-based decoder initialization
- Softmax abundance constraints
- SAD, MSE, MVC, sparsity, and spectral smoothness losses
- Patch-based mixed-precision training to avoid GPU OOM

## Dataset

The main experiment uses the AVIRIS Cuprite hyperspectral dataset.

Original cube:

```text
512 x 614 x 224
