# emb.c - Lightweight Vector Embedding Library

Self-contained lightweight stand-alone vector embedding library based on ggml.

## What It Does

Detailed explanation of features and behavior.

## Public API

```c
kc_emb_t *kc_emb_open(void);
void kc_emb_close(kc_emb_t *ctx);
int kc_emb_exec(kc_emb_t *ctx, const char *input);
```

## CLI Usage

```bash
emb input
```

## Build

POSIX:

```bash
cc -O2 emb.c libemb.c -o emb
```

Windows:

```bash
cl /W4 /TC emb.c libemb.c
```

---

**Author:** KaisarCode

**Email:** <kaisar@kaisarcode.com>

**Website:** [https://kaisarcode.com](https://kaisarcode.com)

**License:** [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

© 2026 KaisarCode
