/**
 * libemb.c - Vector Embedding Library Implementation
 * Summary: Core logic for generating embeddings using ggml.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include <unistd.h>
#endif

#include "emb.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef _WIN32
#include <windows.h>
#endif

#if defined(_WIN32) && !defined(_WIN64)
extern const unsigned char binary_model_gguf_start[] __asm__("_binary_model_gguf_start");
extern const unsigned char binary_model_gguf_end[]   __asm__("_binary_model_gguf_end");
#define model_gguf     ((unsigned char *)binary_model_gguf_start)
#define model_gguf_len ((unsigned int)(binary_model_gguf_end - binary_model_gguf_start))
#else
extern const unsigned char _binary_model_gguf_start[];
extern const unsigned char _binary_model_gguf_end[];
#define model_gguf     ((unsigned char *)_binary_model_gguf_start)
#define model_gguf_len ((unsigned int)(_binary_model_gguf_end - _binary_model_gguf_start))
#endif

typedef struct {
    char *str;
    int id;
} hash_entry;

struct kc_emb_layer {
    struct ggml_tensor *attn_q_w;
    struct ggml_tensor *attn_q_b;
    struct ggml_tensor *attn_k_w;
    struct ggml_tensor *attn_k_b;
    struct ggml_tensor *attn_v_w;
    struct ggml_tensor *attn_v_b;
    struct ggml_tensor *attn_out_w;
    struct ggml_tensor *attn_out_b;
    struct ggml_tensor *attn_norm_w;
    struct ggml_tensor *attn_norm_b;
    struct ggml_tensor *ffn_up_w;
    struct ggml_tensor *ffn_up_b;
    struct ggml_tensor *ffn_down_w;
    struct ggml_tensor *ffn_down_b;
    struct ggml_tensor *layer_norm_w;
    struct ggml_tensor *layer_norm_b;
};

struct kc_emb {
    struct ggml_context *ctx;
    struct gguf_context *gguf;

    int n_vocab;
    int n_embd;
    int n_layer;
    int n_head;
    int n_ctx;
    float layer_norm_eps;

    int cls_token_id;
    int sep_token_id;
    int pad_token_id;
    int unk_token_id;

    char **vocab;
    hash_entry *vocab_hash;
    int vocab_hash_size;

    struct ggml_tensor *token_embd;
    struct ggml_tensor *pos_embd;
    struct ggml_tensor *type_embd;
    struct ggml_tensor *token_embd_norm_w;
    struct ggml_tensor *token_embd_norm_b;

    struct kc_emb_layer *layers;

    void *compute_buf;
    size_t compute_buf_size;
    ggml_backend_t backend;
    ggml_gallocr_t galloc;

    float *out;
};

/**
 * Check if a character is a space (ASCII only).
 * @param c Input character.
 * @return 1 if space, 0 otherwise.
 */
static int kc_isspace(int c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

/**
 * Check if a character is punctuation (ASCII only).
 * @param c Input character.
 * @return 1 if punctuation, 0 otherwise.
 */
static int kc_ispunct(int c) {
    return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
}

/**
 * Convert a character to lowercase (ASCII only).
 * @param c Input character.
 * @return Lowercase character.
 */
static int kc_tolower(int c) {
    return (c >= 'A' && c <= 'Z') ? (c + 32) : c;
}

/**
 * Generate a hash for a given string.
 * @param str Input string.
 * @return Hash value.
 */
static uint32_t hash_str(const char *str) {
    uint32_t h = 5381;
    int c;
    while ((c = *str++)) h = ((h << 5) + h) + (uint8_t)c;
    return h;
}

/**
 * Initialize the vocabulary hash table.
 * @param ctx Context pointer.
 * @return 0 on success, -1 on failure.
 */
static int build_vocab_hash(kc_emb_t *ctx) {
    ctx->vocab_hash_size = ctx->n_vocab * 2 + 1;
    ctx->vocab_hash = (hash_entry *)calloc(ctx->vocab_hash_size, sizeof(hash_entry));

    if (!ctx->vocab_hash) {
        return -1;
    }

    for (int i = 0; i < ctx->n_vocab; i++) {
        uint32_t h = hash_str(ctx->vocab[i]) % ctx->vocab_hash_size;
        while (ctx->vocab_hash[h].str != NULL) {
            h = (h + 1) % ctx->vocab_hash_size;
        }
        ctx->vocab_hash[h].str = ctx->vocab[i];
        ctx->vocab_hash[h].id = i;
    }
    return 0;
}

/**
 * Portable implementation of fmemopen for systems that lack it (e.g. Windows).
 * @param buf Data buffer.
 * @param size Buffer size.
 * @param mode Open mode.
 * @return FILE pointer or NULL on failure.
 */
static FILE *kc_fmemopen(const void *buf, size_t size, const char *mode) {
#if defined(_WIN32) || defined(__ANDROID__)
    FILE *f = tmpfile();
    if (!f) return NULL;
    if (fwrite(buf, 1, size, f) != size) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    return f;
#else
    return fmemopen((void *)buf, size, mode);
#endif
}

/**
 * Null log callback to silence GGML diagnostics.
 * @param level Log level.
 * @param text Log text.
 * @param user_data User data pointer.
 * @return None.
 */
static void kc_ggml_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

/**
 * Search for a token ID in the vocabulary.
 * @param ctx Context pointer.
 * @param str Token string.
 * @return Token ID or -1 if not found.
 */
static int find_token(kc_emb_t *ctx, const char *str) {
    if (!ctx->vocab_hash || ctx->vocab_hash_size == 0) {
        return -1;
    }

    uint32_t h = hash_str(str) % ctx->vocab_hash_size;
    while (ctx->vocab_hash[h].str != NULL) {
        if (strcmp(ctx->vocab_hash[h].str, str) == 0) {
            return ctx->vocab_hash[h].id;
        }
        h = (h + 1) % ctx->vocab_hash_size;
    }
    return -1;
}

/**
 * Retrieve a uint32 value from GGUF metadata with type flexibility.
 * @param ctx GGUF context.
 * @param key Metadata key.
 * @param def Default value.
 * @return Metadata value or default.
 */
static uint32_t get_kv_u32(const struct gguf_context *ctx, const char *key, uint32_t def) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return def;
    enum gguf_type type = gguf_get_kv_type(ctx, id);
    if (type == GGUF_TYPE_UINT32) return gguf_get_val_u32(ctx, id);
    if (type == GGUF_TYPE_INT32) return (uint32_t)gguf_get_val_i32(ctx, id);
    if (type == GGUF_TYPE_UINT64) return (uint32_t)gguf_get_val_u64(ctx, id);
    if (type == GGUF_TYPE_INT64) return (uint32_t)gguf_get_val_i64(ctx, id);
    return def;
}

/**
 * Retrieve the thread count from KC_EMB_THREADS.
 * @return Thread count on success, -1 on invalid.
 */
static int kc_emb_get_threads(void) {
    const char *env = getenv("KC_EMB_THREADS");
    int val = 0;
    int i = 0;

    if (!env) {
#ifdef _WIN32
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        return sysinfo.dwNumberOfProcessors == 0 ? 1 : (int)sysinfo.dwNumberOfProcessors;
#else
        long nproc = sysconf(_SC_NPROCESSORS_ONLN);
        return nproc <= 0 ? 1 : (int)nproc;
#endif
    }

    if (env[0] == '\0') {
        return -1;
    }

    for (i = 0; env[i] != '\0'; i++) {
        if (env[i] < '0' || env[i] > '9') {
            return -1;
        }
        val = val * 10 + (env[i] - '0');
        if (val > 1024) {
            return -1;
        }
    }

    if (val == 0) {
        return -1;
    }

    return val;
}

/**
 * Initialize a new emb context.
 * @param none Unused.
 * @return Context pointer or NULL on failure.
 */
kc_emb_t *kc_emb_open(void) {
    kc_emb_t *ctx = (kc_emb_t *)calloc(1, sizeof(kc_emb_t));
    if (!ctx) {
        return NULL;
    }

    FILE *f = kc_fmemopen(model_gguf, model_gguf_len, "rb");
    if (!f) {
        free(ctx);
        return NULL;
    }

    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = &ctx->ctx,
    };

    ctx->gguf = gguf_init_from_file_ptr(f, params);
    fclose(f);

    if (!ctx->gguf) {
        goto failure;
    }

    {
        size_t data_offset = gguf_get_data_offset(ctx->gguf);
        int n_tensors = gguf_get_n_tensors(ctx->gguf);
        for (int i = 0; i < n_tensors; i++) {
            const char *name = gguf_get_tensor_name(ctx->gguf, i);
            struct ggml_tensor *t = ggml_get_tensor(ctx->ctx, name);
            if (t) {
                t->data = (char *)model_gguf + data_offset + gguf_get_tensor_offset(ctx->gguf, i);
            }
        }
    }

    ctx->n_embd = (int)get_kv_u32(ctx->gguf, "bert.embedding_length", 0);
    ctx->n_layer = (int)get_kv_u32(ctx->gguf, "bert.block_count", 0);
    ctx->n_head = (int)get_kv_u32(ctx->gguf, "bert.attention.head_count", 0);
    ctx->n_ctx = (int)get_kv_u32(ctx->gguf, "bert.context_length", 512);

    if (ctx->n_embd <= 0 || ctx->n_layer <= 0 || ctx->n_head <= 0 || ctx->n_ctx <= 0 || (ctx->n_embd % ctx->n_head) != 0) {
        goto failure;
    }

    int64_t kid;
    kid = gguf_find_key(ctx->gguf, "bert.attention.layer_norm_epsilon");
    ctx->layer_norm_eps = (kid >= 0) ? gguf_get_val_f32(ctx->gguf, kid) : 1e-12f;

    kid = gguf_find_key(ctx->gguf, "tokenizer.ggml.tokens");
    if (kid < 0) {
        goto failure;
    }

    ctx->n_vocab = (int)gguf_get_arr_n(ctx->gguf, kid);
    if (ctx->n_vocab <= 0) {
        goto failure;
    }

    ctx->vocab = (char **)malloc(ctx->n_vocab * sizeof(char *));
    if (!ctx->vocab) {
        goto failure;
    }

    for (int i = 0; i < ctx->n_vocab; i++) {
        ctx->vocab[i] = (char *)gguf_get_arr_str(ctx->gguf, kid, i);
    }

    if (build_vocab_hash(ctx) != 0) {
        if (ctx->vocab_hash) {
            free(ctx->vocab_hash);
            ctx->vocab_hash = NULL;
        }
        free(ctx->vocab);
        ctx->vocab = NULL;
        goto failure;
    }

    ctx->cls_token_id = (int)get_kv_u32(ctx->gguf, "tokenizer.ggml.cls_token_id", 101);
    ctx->sep_token_id = (int)get_kv_u32(ctx->gguf, "tokenizer.ggml.sep_token_id", 102);
    ctx->pad_token_id = (int)get_kv_u32(ctx->gguf, "tokenizer.ggml.pad_token_id", 0);
    ctx->unk_token_id = (int)get_kv_u32(ctx->gguf, "tokenizer.ggml.unknown_token_id", 100);

    ctx->token_embd = ggml_get_tensor(ctx->ctx, "token_embd.weight");
    ctx->pos_embd   = ggml_get_tensor(ctx->ctx, "position_embd.weight");
    ctx->type_embd  = ggml_get_tensor(ctx->ctx, "token_types.weight");
    ctx->token_embd_norm_w = ggml_get_tensor(ctx->ctx, "token_embd_norm.weight");
    ctx->token_embd_norm_b = ggml_get_tensor(ctx->ctx, "token_embd_norm.bias");

    if (!ctx->token_embd || !ctx->pos_embd || !ctx->type_embd || !ctx->token_embd_norm_w || !ctx->token_embd_norm_b) {
        goto failure;
    }

    ctx->layers = (struct kc_emb_layer *)calloc(ctx->n_layer, sizeof(struct kc_emb_layer));
    if (!ctx->layers) {
        goto failure;
    }

    for (int i = 0; i < ctx->n_layer; i++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
        ctx->layers[i].attn_q_w = ggml_get_tensor(ctx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_q.bias", i);
        ctx->layers[i].attn_q_b = ggml_get_tensor(ctx->ctx, name);

        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
        ctx->layers[i].attn_k_w = ggml_get_tensor(ctx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_k.bias", i);
        ctx->layers[i].attn_k_b = ggml_get_tensor(ctx->ctx, name);

        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
        ctx->layers[i].attn_v_w = ggml_get_tensor(ctx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_v.bias", i);
        ctx->layers[i].attn_v_b = ggml_get_tensor(ctx->ctx, name);

        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
        ctx->layers[i].attn_out_w = ggml_get_tensor(ctx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_output.bias", i);
        ctx->layers[i].attn_out_b = ggml_get_tensor(ctx->ctx, name);

        snprintf(name, sizeof(name), "blk.%d.attn_output_norm.weight", i);
        ctx->layers[i].attn_norm_w = ggml_get_tensor(ctx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_output_norm.bias", i);
        ctx->layers[i].attn_norm_b = ggml_get_tensor(ctx->ctx, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
        ctx->layers[i].ffn_up_w = ggml_get_tensor(ctx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.bias", i);
        ctx->layers[i].ffn_up_b = ggml_get_tensor(ctx->ctx, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
        ctx->layers[i].ffn_down_w = ggml_get_tensor(ctx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.bias", i);
        ctx->layers[i].ffn_down_b = ggml_get_tensor(ctx->ctx, name);

        snprintf(name, sizeof(name), "blk.%d.layer_output_norm.weight", i);
        ctx->layers[i].layer_norm_w = ggml_get_tensor(ctx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.layer_output_norm.bias", i);
        ctx->layers[i].layer_norm_b = ggml_get_tensor(ctx->ctx, name);

        if (!ctx->layers[i].attn_q_w || !ctx->layers[i].attn_q_b ||
            !ctx->layers[i].attn_k_w || !ctx->layers[i].attn_k_b ||
            !ctx->layers[i].attn_v_w || !ctx->layers[i].attn_v_b ||
            !ctx->layers[i].attn_out_w || !ctx->layers[i].attn_out_b ||
            !ctx->layers[i].attn_norm_w || !ctx->layers[i].attn_norm_b ||
            !ctx->layers[i].ffn_up_w || !ctx->layers[i].ffn_up_b ||
            !ctx->layers[i].ffn_down_w || !ctx->layers[i].ffn_down_b ||
            !ctx->layers[i].layer_norm_w || !ctx->layers[i].layer_norm_b) {
            goto failure;
        }
    }

    ctx->compute_buf_size = 64 * 1024 * 1024;
    ctx->compute_buf = malloc(ctx->compute_buf_size);
    if (!ctx->compute_buf) {
        goto failure;
    }

    int threads = kc_emb_get_threads();
    if (threads < 0) {
        goto failure;
    }

    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        goto failure;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend, threads);

    ctx->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ctx->galloc) {
        goto failure;
    }

    ctx->out = (float *)malloc(ctx->n_embd * sizeof(float));
    if (!ctx->out) {
        goto failure;
    }

    ggml_log_set(kc_ggml_log_callback, NULL);

    return ctx;

failure:
    kc_emb_close(ctx);
    return NULL;
}

/**
 * Release a emb context.
 * @param ctx Context pointer.
 * @return None.
 */
void kc_emb_close(kc_emb_t *ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->vocab) {
        free(ctx->vocab);
        ctx->vocab = NULL;
    }

    if (ctx->vocab_hash) {
        free(ctx->vocab_hash);
        ctx->vocab_hash = NULL;
    }

    if (ctx->layers) {
        free(ctx->layers);
        ctx->layers = NULL;
    }

    if (ctx->compute_buf) {
        free(ctx->compute_buf);
        ctx->compute_buf = NULL;
    }

    if (ctx->out) {
        free(ctx->out);
        ctx->out = NULL;
    }

    if (ctx->galloc) {
        ggml_gallocr_free(ctx->galloc);
        ctx->galloc = NULL;
    }

    if (ctx->backend) {
        ggml_backend_free(ctx->backend);
        ctx->backend = NULL;
    }

    if (ctx->gguf) {
        gguf_free(ctx->gguf);
        ctx->gguf = NULL;
    }

    if (ctx->ctx) {
        ggml_free(ctx->ctx);
        ctx->ctx = NULL;
    }

    free(ctx);
}

/**
 * Retrieve the embedding dimension.
 * @param ctx Context pointer.
 * @return Dimension size.
 */
int kc_emb_dim(kc_emb_t *ctx) {
    return ctx ? ctx->n_embd : 0;
}

/**
 * Split input text into WordPiece tokens.
 * @param ctx Context pointer.
 * @param input Text input.
 * @param tokens Token ID output array.
 * @param n_tokens Token count output.
 * @return None.
 */
static void wordpiece_tokenize(
    kc_emb_t *ctx,
    const char *input,
    int *tokens,
    int *n_tokens
) {
    *n_tokens = 0;
    if (*n_tokens < ctx->n_ctx) {
        tokens[(*n_tokens)++] = ctx->cls_token_id;
    }

    int len = strlen(input);
    int i = 0;

    while (i < len && *n_tokens < ctx->n_ctx - 1) {
        while (i < len && kc_isspace((uint8_t)input[i])) i++;
        if (i >= len) break;

        if (kc_ispunct((uint8_t)input[i])) {
            char p[2] = { (char)kc_tolower((uint8_t)input[i]), '\0' };
            int id = find_token(ctx, p);
            if (*n_tokens < ctx->n_ctx) {
                tokens[(*n_tokens)++] = id >= 0 ? id : ctx->unk_token_id;
            }
            i++;
            continue;
        }

        int j = i;
        while (j < len && !kc_isspace((uint8_t)input[j]) && !kc_ispunct((uint8_t)input[j])) j++;

        int word_len = j - i;
        char word[128];

        if (word_len >= (int)sizeof(word)) word_len = sizeof(word) - 1;

        for (int k = 0; k < word_len; k++) {
            word[k] = (char)kc_tolower((uint8_t)input[i + k]);
        }
        word[word_len] = '\0';

        int start = 0;

        while (start < word_len && *n_tokens < ctx->n_ctx - 1) {
            int end = word_len;
            int best_id = -1;
            int best_end = -1;

            while (end > start) {
                char subword[128];
                int slen = 0;

                if (start > 0) {
                    subword[0] = '#';
                    subword[1] = '#';
                    slen = 2;
                }

                for (int k = start; k < end; k++) {
                    if (slen >= (int)sizeof(subword) - 1) break;
                    subword[slen++] = word[k];
                }

                subword[slen] = '\0';

                int id = find_token(ctx, subword);
                if (id >= 0) {
                    best_id = id;
                    best_end = end;
                    break;
                }

                end--;
            }

            if (best_id == -1) {
                if (*n_tokens < ctx->n_ctx) {
                    tokens[(*n_tokens)++] = ctx->unk_token_id;
                }
                break;
            }

            if (*n_tokens < ctx->n_ctx) {
                tokens[(*n_tokens)++] = best_id;
            }
            start = best_end;
        }

        i = j;
    }

    if (*n_tokens < ctx->n_ctx) {
        tokens[(*n_tokens)++] = ctx->sep_token_id;
    }
}

/**
 * Generate embeddings for the given input text.
 * @param ctx Context pointer.
 * @param input Text input.
 * @return Pointer to embedding vector or NULL on failure.
 */
float *kc_emb_exec(kc_emb_t *ctx, const char *input) {
    int *tokens = NULL;
    int *pos_data = NULL;
    int *type_data = NULL;
    int n_tokens = 0;
    struct ggml_init_params params;
    struct ggml_context *ctx0 = NULL;
    struct ggml_tensor *inp_tokens = NULL;
    struct ggml_tensor *inp_pos = NULL;
    struct ggml_tensor *inp_type = NULL;
    struct ggml_cgraph *gf = NULL;
    struct ggml_tensor *cur = NULL;
    struct ggml_tensor *pos = NULL;
    struct ggml_tensor *typ = NULL;
    struct ggml_tensor *cls = NULL;
    int d_head = 0;
    const char *prefix = "Represent this sentence for retrieval: ";
    char *norm_input = NULL;
    size_t norm_len = 0;

    if (!ctx || !input) return NULL;

    norm_len = strlen(prefix) + strlen(input) + 1;
    norm_input = (char *)malloc(norm_len);
    if (!norm_input) return NULL;

    snprintf(norm_input, norm_len, "%s%s", prefix, input);

    tokens = (int *)calloc(ctx->n_ctx, sizeof(int));
    pos_data = (int *)calloc(ctx->n_ctx, sizeof(int));
    type_data = (int *)calloc(ctx->n_ctx, sizeof(int));

    if (!tokens || !pos_data || !type_data) {
        goto failure;
    }

    wordpiece_tokenize(ctx, norm_input, tokens, &n_tokens);
    free(norm_input);
    norm_input = NULL;

    if (n_tokens < 2 || n_tokens > ctx->n_ctx) {
        goto failure;
    }

    for (int i = 0; i < n_tokens; i++) pos_data[i] = i;

    params.mem_size   = ctx->compute_buf_size;
    params.mem_buffer = ctx->compute_buf;
    params.no_alloc   = true;

    ctx0 = ggml_init(params);
    if (!ctx0) {
        goto failure;
    }

    inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp_pos    = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp_type   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);

    if (!inp_tokens || !inp_pos || !inp_type) {
        goto failure;
    }

    inp_tokens->data = tokens;
    inp_pos->data = pos_data;
    inp_type->data = type_data;

    gf = ggml_new_graph(ctx0);
    if (!gf) {
        goto failure;
    }

    cur = ggml_get_rows(ctx0, ctx->token_embd, inp_tokens);
    pos = ggml_get_rows(ctx0, ctx->pos_embd, inp_pos);
    typ = ggml_get_rows(ctx0, ctx->type_embd, inp_type);

    if (!cur || !pos || !typ) {
        goto failure;
    }

    cur = ggml_add(ctx0, cur, pos);
    cur = ggml_add(ctx0, cur, typ);
    cur = ggml_norm(ctx0, cur, ctx->layer_norm_eps);
    cur = ggml_add(ctx0, ggml_mul(ctx0, cur, ctx->token_embd_norm_w), ctx->token_embd_norm_b);

    d_head = ctx->n_embd / ctx->n_head;

    for (int il = 0; il < ctx->n_layer; il++) {
        struct ggml_tensor *inp_L = cur;

        struct ggml_tensor *q = ggml_add(ctx0, ggml_mul_mat(ctx0, ctx->layers[il].attn_q_w, cur), ctx->layers[il].attn_q_b);
        struct ggml_tensor *k = ggml_add(ctx0, ggml_mul_mat(ctx0, ctx->layers[il].attn_k_w, cur), ctx->layers[il].attn_k_b);
        struct ggml_tensor *v = ggml_add(ctx0, ggml_mul_mat(ctx0, ctx->layers[il].attn_v_w, cur), ctx->layers[il].attn_v_b);

        q = ggml_reshape_3d(ctx0, q, d_head, ctx->n_head, n_tokens);
        k = ggml_reshape_3d(ctx0, k, d_head, ctx->n_head, n_tokens);
        v = ggml_reshape_3d(ctx0, v, d_head, ctx->n_head, n_tokens);

        q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
        k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));
        v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));

        struct ggml_tensor *kq = ggml_mul_mat(ctx0, q, k);
        kq = ggml_scale_inplace(ctx0, kq, 1.0f / sqrtf((float)d_head));
        kq = ggml_soft_max_inplace(ctx0, kq);

        struct ggml_tensor *kqv = ggml_mul_mat(ctx0, v, kq);
        kqv = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
        kqv = ggml_reshape_2d(ctx0, kqv, ctx->n_embd, n_tokens);

        struct ggml_tensor *attn_out = ggml_add(ctx0, ggml_mul_mat(ctx0, ctx->layers[il].attn_out_w, kqv), ctx->layers[il].attn_out_b);

        cur = ggml_add(ctx0, inp_L, attn_out);
        cur = ggml_norm(ctx0, cur, ctx->layer_norm_eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, ctx->layers[il].attn_norm_w), ctx->layers[il].attn_norm_b);

        struct ggml_tensor *inp_F = cur;
        struct ggml_tensor *ffn = ggml_add(ctx0, ggml_mul_mat(ctx0, ctx->layers[il].ffn_up_w, cur), ctx->layers[il].ffn_up_b);
        ffn = ggml_gelu(ctx0, ffn);
        ffn = ggml_add(ctx0, ggml_mul_mat(ctx0, ctx->layers[il].ffn_down_w, ffn), ctx->layers[il].ffn_down_b);

        cur = ggml_add(ctx0, inp_F, ffn);
        cur = ggml_norm(ctx0, cur, ctx->layer_norm_eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, ctx->layers[il].layer_norm_w), ctx->layers[il].layer_norm_b);
    }

    if (cur->ne[0] != ctx->n_embd) {
        goto failure;
    }

    cls = ggml_view_2d(ctx0, cur, ctx->n_embd, 1, cur->nb[1], 0);
    cls = ggml_cont(ctx0, cls);

    ggml_build_forward_expand(gf, cls);

    if (!ggml_gallocr_alloc_graph(ctx->galloc, gf)) {
        goto failure;
    }

    memcpy(inp_tokens->data, tokens, n_tokens * sizeof(int));
    memcpy(inp_pos->data, pos_data, n_tokens * sizeof(int));
    memset(inp_type->data, 0, n_tokens * sizeof(int));

    ggml_backend_graph_compute(ctx->backend, gf);

    if (!cls->data || cls->ne[0] != ctx->n_embd) {
        goto failure;
    }

    memcpy(ctx->out, cls->data, ctx->n_embd * sizeof(float));

    ggml_free(ctx0);
    free(tokens);
    free(pos_data);
    free(type_data);
    return ctx->out;

failure:
    if (ctx0) ggml_free(ctx0);
    if (tokens) free(tokens);
    if (pos_data) free(pos_data);
    if (type_data) free(type_data);
    if (norm_input) free(norm_input);
    return NULL;
}
