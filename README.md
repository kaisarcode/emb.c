# emb.c - Lightweight Vector Embedding Library

A minimalist C library for vector embeddings using BERT-based models. Optimized for CPU inference with zero-copy weight mapping.

---

## Performance
The engine is optimized for local environments, achieving ~0.026s latency per inference on standard hardware.

---

## Quick Start

### 1. Build
Requires a C++ compiler and CMake 3.14+.
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release
```
*The `emb` binary will be generated directly in the root directory.*

### 2. Usage
```c
#include "emb.h"

// Initialize context
kc_emb_t *ctx = kc_emb_open();

// Generate embeddings
kc_emb_exec(ctx, "The cat is green");

// Clean up
kc_emb_close(ctx);
```

---

## Features
- **Native SIMD**: Leverages AVX2, AVX512, and NEON kernels.
- **Zero-Copy**: Maps model weights directly from memory.
- **Self-Contained**: Includes the `bge-small-en-v1.5` model.

---

## Changing the Model
To use a different GGUF model:
1. Generate `model.c`: `xxd -i your_model.gguf > model.c`
2. Re-run the build command.

---

**Author:** KaisarCode

**Email:** [kaisar@kaisarcode.com](mailto:kaisar@kaisarcode.com)

**Website:** https://kaisarcode.com

**License:** https://www.gnu.org/licenses/gpl-3.0.html

© 2026 KaisarCode
