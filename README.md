# emb.c - Lightweight Vector Embedding Library

Self-contained vector embedding library with a built-in model. Designed for easy integration into minimal C/C++ projects.

## What It Does

This project is a single-unit vector embedding engine. By default, it includes the **bge-small-en-v1.5-q4_k_m.gguf** model, which is distributed as a C source file (`model.c`). When compiled into an object file, the model occupies ~24MB within the final binary.

**Note on Performance:** This implementation does not include GPU acceleration. Given the small size of the base model, CPU inference is extremely fast and efficient. For more complex projects requiring larger models, modifications to support GPU backends would be necessary.

## Public API

```c
kc_emb_t *kc_emb_open(void);
void      kc_emb_close(kc_emb_t *ctx);
int       kc_emb_exec(kc_emb_t *ctx, const char *input);
```

## Prerequisites

To build the library, you need:
- A standard C/C++ compiler (`gcc`/`g++` or `cl`)

## Build Instructions

To keep development fast, we compile the large model data separately and build `ggml` as a static library so they only link during the final stage. 
This 3-stage process allows you to iterate on your code instantly without recompiling massive files.

### Build (Linux & Windows)

The project uses a unified CMake build system that automatically optimizes the engine for your local hardware.

```bash
# 1. Configure the project
# This will detect your CPU features (AVX2, AVX512, etc.)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. Build the engine
cmake --build build --config Release
```

The resulting binary will be located in `build/emb` (Linux) or `build/bin/Release/emb.exe` (Windows).

---

## Changing the Model

If you wish to use a different GGUF-compatible model, you can replace the built-in one by generating a new `model.c`:

1. Download your desired `.gguf` model.
2. Run the following command to convert the binary into C source:
```bash
xxd -i your_model.gguf > model.c
```
3. Recompile the `model.o` (Step 1 of the Build Instructions).

**Note:** Generating `model.c` from a binary creates a very large text file (~154MB for a 24MB model). This is normal and only affects compilation time of that specific file.

---

**Author:** KaisarCode

**Email:** [kaisar@kaisarcode.com](mailto:kaisar@kaisarcode.com)

**Website:** https://kaisarcode.com

**License:** https://www.gnu.org/licenses/gpl-3.0.html

© 2026 KaisarCode
