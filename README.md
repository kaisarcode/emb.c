# emb.c — Lightweight Vector Embedding Library

A minimalist C library for generating vector embeddings using `bge-small-en-v1.5`. CPU inference, zero-copy weight mapping, model embedded at compile time.

## File Layout

```
emb.c/
├── src/
│   ├── emb.c          CLI entry point (main)
│   ├── libemb.c       Core library implementation
│   └── emb.h          Public API header
├── lib/
│   ├── ggml/          Vendored GGML dependency
│   └── model.gguf     Embedded model (Git LFS)
├── bin/               Compiled artifacts (committed, Git LFS)
│   ├── x86_64/{linux,windows}
│   ├── i686/{linux,windows}
│   ├── aarch64/{linux,android}
│   ├── armv7/{linux,android}
│   ├── armv7hf/linux
│   ├── riscv64/linux
│   ├── powerpc64le/linux
│   ├── mips/linux  mipsel/linux  mips64el/linux
│   ├── s390x/linux
│   └── loongarch64/linux
├── CMakeLists.txt
├── Makefile
├── test.sh
└── README.md
```

## Build

```bash
make all              # all 16 targets
make x86_64/linux
make x86_64/windows
make i686/linux
make i686/windows
make aarch64/linux
make aarch64/android
make armv7/linux
make armv7/android
make armv7hf/linux
make riscv64/linux
make powerpc64le/linux
make mips/linux
make mipsel/linux
make mips64el/linux
make s390x/linux
make loongarch64/linux
make clean
```

Each target produces under `bin/{arch}/{platform}/`:
- `libemb.a` — static library
- `libemb.so` / `libemb.dll` / `libemb.dll.a` — shared library and import lib
- `emb` / `emb.exe` — CLI executable

## CLI

```bash
# Single input via arguments
emb "The cat is green"

# Multiple inputs via stdin (one embedding per line)
cat sentences.txt | emb

# Redirected input (one embedding per line)
emb < input.txt
```

The CLI processes input from arguments or `stdin`. When using arguments, they are joined into a single input string. When using `stdin` (piped or redirected), the tool processes input line-by-line, outputting one vector per line. This is the most efficient way to process multiple strings as the model is only loaded once.

The CLI outputs each vector embedding as a single line of space-separated floating point numbers to stdout. Errors are written to stderr.

## Usage

```c
#include "emb.h"

kc_emb_t *ctx = kc_emb_open();
int dim = kc_emb_dim(ctx);
float out[384];
kc_emb_exec(ctx, "The cat is green", out);
kc_emb_close(ctx);
```

## Changing the Model

Replace `lib/model.gguf` with any GGUF BERT-compatible model and rebuild.

## Lifecycle

- `kc_emb_open()` — allocates and returns a context owned by the caller. It
    prepares one GGML inference context backed by the embedded model binary.
- `kc_emb_close()` — shuts down the worker and releases all resources. Must
    not be called while `kc_emb_exec()` is active on any thread.
- `kc_emb_exec()` — dispatches to the prepared worker and blocks until the
    result is ready. Each request uses up to four CPU threads. Multiple
    callers on the same context are serialized.

---

**Author:** KaisarCode

**Email:** <kaisarcode@gmail.com>

**Website:** [https://kaisarcode.com](https://kaisarcode.com)

**License:** [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

© 2026 KaisarCode
