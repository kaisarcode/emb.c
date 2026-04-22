/**
 * emb.h - Vector Embedding Library Public API
 * Summary: Public interface for the ggml-based embedding library.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef KC_EMB_H
#define KC_EMB_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_emb kc_emb_t;

#define KC_EMB_OK      0
#define KC_EMB_ERROR  -1

/**
 * Initialize a new emb context.
 * @param none Unused.
 * @return Context pointer or NULL on failure.
 */
kc_emb_t *kc_emb_open(void);

/**
 * Release a emb context.
 * @param ctx Context pointer.
 * @return None.
 */
void kc_emb_close(kc_emb_t *ctx);

/**
 * Retrieve the embedding dimension.
 * @param ctx Context pointer.
 * @return Dimension size.
 */
int kc_emb_dim(kc_emb_t *ctx);

/**
 * Execute a core emb operation.
 * @param ctx Context pointer.
 * @param input Operation input.
 * @return Pointer to embedding vector or NULL on failure.
 *
 * Note:
 * - The returned pointer is owned by the context.
 * - It remains valid until the next call to kc_emb_exec()
 *   or until kc_emb_close() is called.
 * - The caller must NOT free this pointer.
 */
float *kc_emb_exec(kc_emb_t *ctx, const char *input);

#ifdef __cplusplus
}
#endif

#endif
