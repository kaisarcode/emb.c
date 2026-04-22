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

To keep development fast, we compile the large model data separately and build `ggml` as a static library so they only link during the final stage. 
This 3-stage process allows you to iterate on your code instantly without recompiling massive files.

### POSIX

```bash
# 1. Compile the model data once (creates a ~24MB object)
gcc -O2 -c model.c -o model.o

# 2. Compile GGML into a static library (libggml.a)
# Note: we use GGML_CPU_GENERIC for maximum portability. For performance, use native CMake.
GGML_FLAGS="-O2 -D_GNU_SOURCE -DGGML_VERSION=\"1\" -DGGML_COMMIT=\"custom\" -DGGML_CPU_GENERIC -Iggml/include -Iggml/src -Iggml/src/ggml-cpu"

# Compile C sources
gcc $GGML_FLAGS -c ggml/src/ggml.c -o ggml.o
gcc $GGML_FLAGS -c ggml/src/ggml-quants.c -o ggml-quants.o
gcc $GGML_FLAGS -c ggml/src/ggml-alloc.c -o ggml-alloc.o
gcc $GGML_FLAGS -c ggml/src/ggml-cpu/ggml-cpu.c -o ggml-cpu-c.o
gcc $GGML_FLAGS -c ggml/src/ggml-cpu/quants.c -o quants-cpu.o

# Compile C++ sources
g++ $GGML_FLAGS -c ggml/src/ggml.cpp -o ggml-cpp.o
g++ $GGML_FLAGS -c ggml/src/ggml-threading.cpp -o ggml-threading.o
g++ $GGML_FLAGS -c ggml/src/gguf.cpp -o gguf.o
g++ $GGML_FLAGS -c ggml/src/ggml-backend.cpp -o ggml-backend.o
g++ $GGML_FLAGS -c ggml/src/ggml-backend-meta.cpp -o ggml-backend-meta.o
g++ $GGML_FLAGS -c ggml/src/ggml-backend-reg.cpp -o ggml-backend-reg.o
g++ $GGML_FLAGS -c ggml/src/ggml-cpu/ggml-cpu.cpp -o ggml-cpu-cpp.o
g++ $GGML_FLAGS -c ggml/src/ggml-cpu/ops.cpp -o ops.o
g++ $GGML_FLAGS -c ggml/src/ggml-cpu/unary-ops.cpp -o unary-ops.o
g++ $GGML_FLAGS -c ggml/src/ggml-cpu/binary-ops.cpp -o binary-ops.o
g++ $GGML_FLAGS -c ggml/src/ggml-cpu/vec.cpp -o vec.o
g++ $GGML_FLAGS -c ggml/src/ggml-cpu/traits.cpp -o traits.o
g++ $GGML_FLAGS -c ggml/src/ggml-cpu/repack.cpp -o repack.o

# Create archive and clean up objects
ar rcs libggml.a ggml.o ggml-quants.o ggml-alloc.o ggml-cpu-c.o quants-cpu.o ggml-cpp.o ggml-threading.o gguf.o ggml-backend.o ggml-backend-meta.o ggml-backend-reg.o ggml-cpu-cpp.o ops.o unary-ops.o binary-ops.o vec.o traits.o repack.o
rm -f ggml.o ggml-quants.o ggml-alloc.o ggml-cpu-c.o quants-cpu.o ggml-cpp.o ggml-threading.o gguf.o ggml-backend.o ggml-backend-meta.o ggml-backend-reg.o ggml-cpu-cpp.o ops.o unary-ops.o binary-ops.o vec.o traits.o repack.o

# 3. Build and link the engine
g++ -O2 -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu emb.c libemb.c model.o libggml.a -o emb -lm -lpthread
```

### Windows

For Windows, it is highly recommended to use the official CMake build process for GGML to generate `ggml.lib`, then link it.

```batch
:: 1. Compile the model data once
cl /c /O2 model.c /Fo:model.obj

:: 2. Build GGML (Using CMake)
cd ggml
cmake -B build -A x64
cmake --build build --config Release
cd ..

:: 3. Build and link your library
cl /O2 /W4 /I. /Iggml/include /Iggml/src /Iggml/src/ggml-cpu ^
    emb.c libemb.c model.obj ^
    ggml/build/src/Release/ggml.lib ^
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
