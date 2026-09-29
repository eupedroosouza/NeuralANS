# NeuralANS – Neural Network Tensor Compression with Asymmetric Numeral Systems (ANS)

## Overview

**NeuralANS** is a research-oriented tool forked of [StaticBAC](https://github.com/Jiovana/StaticBAC) (a NN tensor compression with Static Binary Arithmetic Coding (StaticBAC)).
The main target is same of StaticBAC: compress already quantized models in INT8 (preferrably) also  supporting  INT4, to INT32, in steps of 4. But, now using the **Asymmetric Numeral Systems** (ANS) coding, more specific the implementations: range-ANS (rANS) and tabled-ANS (tANS).

The research pipeline:

1. Extract tensors from a trained model (PyTorch / HuggingFace / Torchvision)
2. Quantize weights and biases to fixed-point representations
3. Encode tensors based on static context with ANS
4. Optionally decode to reconstruct the model

This framework is designed for:

- Studying entropy coding efficiency on NN weights
- Evaluating compression vs. distortion trade-offs
- Hardware-oriented research (e.g., CABAC-like implementations)
- Studying and comparing the modern ANS compression algorithm to BAC

---

## Features

- **Asymmetric Numeral Systems (ANS)** Coding
- Separate **encode / decode modes**
- Metadata-driven reconstruction
- Tensor-level statistics and performance reporting

---

## Repository Structure

```text
NeuralANS/
│
├── source/          # C++ source code (encoder/decoder)
├── python/          # Python preprocessing (tables create & model export & quantization)
└── README.md
```
Main file is **source/Entrypoint.cpp**.

## Build

### Requirements

#### C++ (Core codec)

- C++17 compatible compiler
- CMake ≥ 3.13

#### Python (Model export)

- Python ≥ 3.10
- Required packages: See [requirements.txt](python/requirements.txt).

### Building

```bash
git clone <repo_url>
cd NeuralANS
mkdir build
cd build
cmake ..
cmake --build .
```

This generates the executable (e.g., NeuralANS).

### Python: Export and Quantize Model

Use the provided script to extract tensors and generate:

Binary tensor files (.bin)
Metadata file (tensor.meta)

Example

```bash
python create_meta.py \
    --model resnet50 \
    --source torchvision \
    --weights ResNet50_Weights.DEFAULT \
    --out_dir ./models/resnet50
```

Output:
```text
models/resnet50/
├── binaries/
│   ├── layer1.weight.bin
│   ├── layer1.bias.bin
│   └── ...
└── tensor.meta
```

## Running the Codec
[NeuralANS.h](source/NeuralANS.h)
The codec supports:

- Encoding only
- Decoding only
- Full encode → decode pipeline

**Only Encode**

```shell
./NeuralANS \
    --encode \
    --binaries ./models/resnet50/binaries \
    --meta ./models/resnet50/tensor.meta \
    --bitstream output.bin
```

**Only Decode**

```shell
./NeuralANS \
    --decode \
    --bitstream output.bin \
    --out_dir ./decoded_model
```

**Encode + Decode**

```shell
./NeuralANS \
    --encode --decode \
    --binaries ./models/resnet50/binaries \
    --meta ./models/resnet50/tensor.meta \
    --bitstream output.bin \
    --out_dir ./decoded_model
```

### Metadata Format

The tensor.meta file describes all tensors:

```text
numTensors N

id name type bitwidth dims shape... qstep
```
Example:\
``0 layer1.weight weight 8 4 64 3 7 7 0.0231``

This enables:

- Correct reconstruction of tensor shapes
- Proper dequantization
- Mapping back to model structure

### Quantization Strategy

**Weights** → 8-bit optimal uniform quantization\
**Bias / other tensors** → 12-bit quantization\
**Small tensors (<32 elements)** → higher precision\
**Buffers** → stored as raw int32 (no quantization)

Quantization step (qstep) is optimized via:
- Golden-section search
- Mean squared error (MSE) minimization
- Performance Metrics

## The tool reports:

- Encoding time
- Decoding time
- Compression ratio
- Entropy (bits/symbol)
- Throughput (MB/s)

### Notes

Tensor names are preserved to simplify reconstruction\
Minimal filename sanitization is recommended (/ and \ replaced) - in create_meta\
Decoding reconstructs quantized tensors (not original float values)

### Future Work

* Might update static RLPS
* Exploit tensor semantics to improve contexts
* Parallel chunk processing to speed up encoder, also important for hardware implementation.
