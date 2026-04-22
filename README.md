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

## Build Instructions

To keep development fast, we compile the large model data separately so it only links during the final stage.

### POSIX (Linux, macOS, MinGW)

```bash
# 1. Compile the model data once (creates a ~24MB object)
g++ -c -O2 model.c -o model.o

# 2. Build the engine (fast)
g++ -O2 -Wno-pragma-once \
    -DGGML_VERSION=\"1\" -DGGML_COMMIT=\"custom\" \
    -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu \
    emb.c libemb.c model.o \
    ggml/src/ggml.c \
    ggml/src/ggml-quants.c \
    ggml/src/ggml-alloc.c \
    ggml/src/ggml-backend.cpp \
    ggml/src/ggml-cpu/ggml-cpu.c \
    -o emb -lm -lpthread
```

### Windows (MSVC)

```batch
:: 1. Compile the model data once
cl /c /O2 model.c /Fo:model.obj

:: 2. Build and link
cl /O2 /W4 /I. /Iggml/include /Iggml/src /Iggml/src/ggml-cpu ^
    -DGGML_VERSION="1" -DGGML_COMMIT="custom" ^
    emb.c libemb.c model.obj ^
    ggml/src/ggml.c ^
    ggml/src/ggml-quants.c ^
    ggml/src/ggml-alloc.c ^
    ggml/src/ggml-backend.cpp ^
    ggml/src/ggml-cpu/ggml-cpu.c ^
    /Fe:emb.exe
```

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

**Website:** [https://kaisarcode.com](https://kaisarcode.com)

**License:** [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

© 2026 KaisarCode
