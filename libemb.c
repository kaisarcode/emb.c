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
#endif

#include "emb.h"

#include <stdio.h>
#include <stdlib.h>
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

#define MAX_LAYERS 32
#define MAX_TOKENS 512

extern unsigned char model_gguf[];
extern unsigned int model_gguf_len;

typedef struct {
    char *str;
    int id;
} hash_entry;

struct kc_emb {
    struct ggml_context *ctx;
    struct gguf_context *gguf;

    int n_vocab;
    int n_embd;
    int n_layer;
    int n_head;
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

    struct {
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
    } layers[MAX_LAYERS];
};

/**
 * Generate a hash for a given string.
 * @param str Input string.
 * @return Hash value.
 */
static uint32_t hash_str(const char *str) {
    uint32_t h = 5381;
    int c;
    while ((c = *str++)) h = ((h << 5) + h) + c;
    return h;
}

/**
 * Initialize the vocabulary hash table.
 * @param ctx Context pointer.
 * @return None.
 */
static void build_vocab_hash(kc_emb_t *ctx) {
    ctx->vocab_hash_size = ctx->n_vocab * 2 + 1;
    ctx->vocab_hash = (hash_entry *)calloc(ctx->vocab_hash_size, sizeof(hash_entry));

    for (int i = 0; i < ctx->n_vocab; i++) {
        uint32_t h = hash_str(ctx->vocab[i]) % ctx->vocab_hash_size;
        while (ctx->vocab_hash[h].str != NULL) {
            h = (h + 1) % ctx->vocab_hash_size;
        }
        ctx->vocab_hash[h].str = ctx->vocab[i];
        ctx->vocab_hash[h].id = i;
    }
}

/**
 * Search for a token ID in the vocabulary.
 * @param ctx Context pointer.
 * @param str Token string.
 * @return Token ID or -1 if not found.
 */
static int find_token(kc_emb_t *ctx, const char *str) {
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
 * Initialize a new emb context.
 * @param none Unused.
 * @return Context pointer or NULL on failure.
 */
kc_emb_t *kc_emb_open(void) {
    kc_emb_t *ctx = (kc_emb_t *)calloc(1, sizeof(kc_emb_t));
    if (!ctx) {
        return NULL;
    }

    FILE *f = fmemopen(model_gguf, model_gguf_len, "rb");
    if (!f) {
        free(ctx);
        return NULL;
    }

    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = &ctx->ctx,
    };

    ctx->gguf = gguf_init_from_file_ptr(f, params);
    fclose(f);

    if (!ctx->gguf) {
        free(ctx);
        return NULL;
    }

    ctx->n_embd = (int)get_kv_u32(ctx->gguf, "bert.embedding_length", 0);
    ctx->n_layer = (int)get_kv_u32(ctx->gguf, "bert.block_count", 0);
    ctx->n_head = (int)get_kv_u32(ctx->gguf, "bert.attention.head_count", 0);

    int64_t kid;
    kid = gguf_find_key(ctx->gguf, "bert.attention.layer_norm_epsilon");
    ctx->layer_norm_eps = (kid >= 0) ? gguf_get_val_f32(ctx->gguf, kid) : 1e-12f;

    kid = gguf_find_key(ctx->gguf, "tokenizer.ggml.tokens");
    if (kid >= 0) {
        ctx->n_vocab = (int)gguf_get_arr_n(ctx->gguf, kid);
        ctx->vocab = (char **)malloc(ctx->n_vocab * sizeof(char *));
        for (int i = 0; i < ctx->n_vocab; i++) {
            ctx->vocab[i] = strdup(gguf_get_arr_str(ctx->gguf, kid, i));
        }
        build_vocab_hash(ctx);
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

    for (int i = 0; i < ctx->n_layer && i < MAX_LAYERS; i++) {
        char name[64];
        sprintf(name, "blk.%d.attn_q.weight", i);
        ctx->layers[i].attn_q_w = ggml_get_tensor(ctx->ctx, name);
        sprintf(name, "blk.%d.attn_q.bias", i);
        ctx->layers[i].attn_q_b = ggml_get_tensor(ctx->ctx, name);

        sprintf(name, "blk.%d.attn_k.weight", i);
        ctx->layers[i].attn_k_w = ggml_get_tensor(ctx->ctx, name);
        sprintf(name, "blk.%d.attn_k.bias", i);
        ctx->layers[i].attn_k_b = ggml_get_tensor(ctx->ctx, name);

        sprintf(name, "blk.%d.attn_v.weight", i);
        ctx->layers[i].attn_v_w = ggml_get_tensor(ctx->ctx, name);
        sprintf(name, "blk.%d.attn_v.bias", i);
        ctx->layers[i].attn_v_b = ggml_get_tensor(ctx->ctx, name);

        sprintf(name, "blk.%d.attn_output.weight", i);
        ctx->layers[i].attn_out_w = ggml_get_tensor(ctx->ctx, name);
        sprintf(name, "blk.%d.attn_output.bias", i);
        ctx->layers[i].attn_out_b = ggml_get_tensor(ctx->ctx, name);

        sprintf(name, "blk.%d.attn_output_norm.weight", i);
        ctx->layers[i].attn_norm_w = ggml_get_tensor(ctx->ctx, name);
        sprintf(name, "blk.%d.attn_output_norm.bias", i);
        ctx->layers[i].attn_norm_b = ggml_get_tensor(ctx->ctx, name);

        sprintf(name, "blk.%d.ffn_up.weight", i);
        ctx->layers[i].ffn_up_w = ggml_get_tensor(ctx->ctx, name);
        sprintf(name, "blk.%d.ffn_up.bias", i);
        ctx->layers[i].ffn_up_b = ggml_get_tensor(ctx->ctx, name);

        sprintf(name, "blk.%d.ffn_down.weight", i);
        ctx->layers[i].ffn_down_w = ggml_get_tensor(ctx->ctx, name);
        sprintf(name, "blk.%d.ffn_down.bias", i);
        ctx->layers[i].ffn_down_b = ggml_get_tensor(ctx->ctx, name);

        sprintf(name, "blk.%d.layer_output_norm.weight", i);
        ctx->layers[i].layer_norm_w = ggml_get_tensor(ctx->ctx, name);
        sprintf(name, "blk.%d.layer_output_norm.bias", i);
        ctx->layers[i].layer_norm_b = ggml_get_tensor(ctx->ctx, name);
    }

    return ctx;
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
        for (int i = 0; i < ctx->n_vocab; i++) {
            free(ctx->vocab[i]);
        }
        free(ctx->vocab);
    }

    if (ctx->vocab_hash) {
        free(ctx->vocab_hash);
    }

    if (ctx->gguf) {
        gguf_free(ctx->gguf);
    }

    if (ctx->ctx) {
        ggml_free(ctx->ctx);
    }

    free(ctx);
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
    tokens[(*n_tokens)++] = ctx->cls_token_id;

    int len = strlen(input);
    int i = 0;

    while (i < len && *n_tokens < MAX_TOKENS - 1) {
        while (i < len && isspace(input[i])) i++;
        if (i >= len) break;

        if (ispunct(input[i])) {
            char p[2] = { (char)tolower(input[i]), '\0' };
            int id = find_token(ctx, p);
            tokens[(*n_tokens)++] = id >= 0 ? id : ctx->unk_token_id;
            i++;
            continue;
        }

        int j = i;
        while (j < len && !isspace(input[j]) && !ispunct(input[j])) j++;

        int word_len = j - i;
        char word[128];

        if (word_len >= (int)sizeof(word)) word_len = sizeof(word) - 1;

        for (int k = 0; k < word_len; k++) {
            word[k] = tolower(input[i + k]);
        }
        word[word_len] = '\0';

        int start = 0;

        while (start < word_len && *n_tokens < MAX_TOKENS - 1) {
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
                tokens[(*n_tokens)++] = ctx->unk_token_id;
                break;
            }

            tokens[(*n_tokens)++] = best_id;
            start = best_end;
        }

        i = j;
    }

    if (*n_tokens < MAX_TOKENS) {
        tokens[(*n_tokens)++] = ctx->sep_token_id;
    }
}

/**
 * Generate embeddings for the given input text.
 * @param ctx Context pointer.
 * @param input Text input.
 * @return Status code.
 */
int kc_emb_exec(kc_emb_t *ctx, const char *input) {
    if (!ctx || !input) return KC_EMB_ERROR;

    int tokens[MAX_TOKENS];
    int n_tokens;
    wordpiece_tokenize(ctx, input, tokens, &n_tokens);

    size_t buf_size = 1024 * 1024 * 64;
    void *buf = malloc(buf_size);
    if (!buf) return KC_EMB_ERROR;

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf,
        .no_alloc   = false,
    };

    struct ggml_context *ctx0 = ggml_init(params);
    struct ggml_cgraph *gf = ggml_new_graph(ctx0);

    struct ggml_tensor *inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    memcpy(inp_tokens->data, tokens, n_tokens * sizeof(int));

    struct ggml_tensor *inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    int *pos_data = (int *)inp_pos->data;
    for (int i = 0; i < n_tokens; i++) pos_data[i] = i;

    struct ggml_tensor *inp_type = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    memset(inp_type->data, 0, n_tokens * sizeof(int));

    struct ggml_tensor *cur = ggml_get_rows(ctx0, ctx->token_embd, inp_tokens);
    struct ggml_tensor *pos = ggml_get_rows(ctx0, ctx->pos_embd, inp_pos);
    struct ggml_tensor *typ = ggml_get_rows(ctx0, ctx->type_embd, inp_type);

    cur = ggml_add(ctx0, cur, pos);
    cur = ggml_add(ctx0, cur, typ);

    cur = ggml_norm(ctx0, cur, ctx->layer_norm_eps);
    cur = ggml_add(ctx0,
        ggml_mul(ctx0, cur, ctx->token_embd_norm_w),
        ctx->token_embd_norm_b);

    int d_head = ctx->n_embd / ctx->n_head;

    for (int il = 0; il < ctx->n_layer; il++) {
        struct ggml_tensor *inp_L = cur;

        struct ggml_tensor *q = ggml_add(ctx0,
            ggml_mul_mat(ctx0, ctx->layers[il].attn_q_w, cur),
            ctx->layers[il].attn_q_b);

        struct ggml_tensor *k = ggml_add(ctx0,
            ggml_mul_mat(ctx0, ctx->layers[il].attn_k_w, cur),
            ctx->layers[il].attn_k_b);

        struct ggml_tensor *v = ggml_add(ctx0,
            ggml_mul_mat(ctx0, ctx->layers[il].attn_v_w, cur),
            ctx->layers[il].attn_v_b);

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

        struct ggml_tensor *attn_out = ggml_add(ctx0,
            ggml_mul_mat(ctx0, ctx->layers[il].attn_out_w, kqv),
            ctx->layers[il].attn_out_b);

        cur = ggml_add(ctx0, inp_L, attn_out);

        cur = ggml_norm(ctx0, cur, ctx->layer_norm_eps);
        cur = ggml_add(ctx0,
            ggml_mul(ctx0, cur, ctx->layers[il].attn_norm_w),
            ctx->layers[il].attn_norm_b);

        struct ggml_tensor *inp_F = cur;

        struct ggml_tensor *ffn = ggml_add(ctx0,
            ggml_mul_mat(ctx0, ctx->layers[il].ffn_up_w, cur),
            ctx->layers[il].ffn_up_b);

        ffn = ggml_gelu(ctx0, ffn);

        ffn = ggml_add(ctx0,
            ggml_mul_mat(ctx0, ctx->layers[il].ffn_down_w, ffn),
            ctx->layers[il].ffn_down_b);

        cur = ggml_add(ctx0, inp_F, ffn);

        cur = ggml_norm(ctx0, cur, ctx->layer_norm_eps);
        cur = ggml_add(ctx0,
            ggml_mul(ctx0, cur, ctx->layers[il].layer_norm_w),
            ctx->layers[il].layer_norm_b);
    }

    ggml_build_forward_expand(gf, cur);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    ggml_backend_graph_compute(backend, gf);

    float *out_data = (float *)cur->data;

    for (int i = 0; i < ctx->n_embd; i++) {
        printf("%f%s", out_data[i], i == ctx->n_embd - 1 ? "" : " ");
    }
    printf("\n");

    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(ctx0);
    free(buf);

    return KC_EMB_OK;
}
