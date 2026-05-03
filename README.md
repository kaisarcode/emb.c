# emb.c - Vector Embedding Library

`emb.c` is a portable C library and CLI for generating vector embeddings from text using a GGML-based model. It is designed as a composable primitive.

---

## CLI

Generate vector embeddings from command-line arguments or standard input.

### Examples

Single sentence embedding:

```bash
./bin/x86_64/linux/emb "The quick brown fox"
```

Batch processing via standard input:

```bash
echo "The quick brown fox" | ./bin/x86_64/linux/emb
cat sentences.txt | ./bin/x86_64/linux/emb
```

---

### Parameters

| Flag | Description |
| :--- | :--- |
| `-h`, `--help` | Show help and usage |
| `-v`, `--version` | Show version |

---

### Output

Results are printed as space-separated floats, one line per input text:

```
0.123456 0.234567 ... 0.345678
```

---

## Public API

```c
#include "emb.h"

kc_emb_t *ctx = kc_emb_open();
int dim = kc_emb_dim(ctx);
float *vec = malloc(dim * sizeof(float));

kc_emb_exec(ctx, "The quick brown fox", vec);

kc_emb_close(ctx);
```

---

## Lifecycle

- `kc_emb_open()` - allocates and prepares a new embedding context.
- `kc_emb_exec()` - generates an embedding for the given text. Multiple calls on the same context are serialized.
- `kc_emb_close()` - releases the context and all associated resources.

---

## Build

```bash
make
make all
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

Artifacts are generated under:

```
bin/{arch}/{platform}/
```

---

**Author:** KaisarCode

**Email:** <kaisar@kaisarcode.com>

**Website:** [https://kaisarcode.com](https://kaisarcode.com)

**License:** [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

© 2026 KaisarCode
