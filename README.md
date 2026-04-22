# emb.c - High-Performance Vector Embedding Library

A minimalist, ultra-fast C library for generating vector embeddings using BERT-based models. Specialized for `bge-small-en-v1.5`, it achieves sub-30ms latency through native hardware optimization and zero-copy weight mapping.

## Key Features

- **Extreme Performance**: Optimized SIMD kernels (AVX2, AVX512, NEON) for record-breaking inference speed.
- **Zero-Copy Architecture**: Models are mapped directly from memory, eliminating redundant allocations and I/O overhead.
- **Self-Contained**: Includes the `bge-small` model embedded as a C source for seamless distribution.
- **KCS Compliant**: Adheres to KaisarCode Standards for maximum code quality and maintainability.
- **Unified Build**: Single CMake workflow for Linux and Windows.

## Performance Benchmark

In local environments, `emb.c` consistently outperforms general-purpose engines:
- **emb.c**: ~0.026s (Real)
- **General Engines**: ~0.080s (Real)

## Prerequisites

To build the library, you need:
- A standard C/C++ compiler (`gcc`/`g++` or `cl`)
- CMake (version 3.14 or higher)

## Build Instructions

```bash
# Build the engine (Linux & Windows)
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release
```

The resulting binary `emb` (or `emb.exe`) will be generated directly in the project root.

## Public API

```c
/* Initialize the embedding context */
kc_emb_t *kc_emb_open(void);

/* Execute inference and return 0 on success */
int kc_emb_exec(kc_emb_t *ctx, const char *input);

/* Clean up resources */
void kc_emb_close(kc_emb_t *ctx);
```

## Changing the Model

To replace the built-in model, generate a new `model.c` from any GGUF file:

1. Download a `.gguf` model.
2. Convert it using `xxd`:
   ```bash
   xxd -i your_model.gguf > model.c
   ```
3. Re-run the build instructions.

---
KaisarCode - High Performance Computing
