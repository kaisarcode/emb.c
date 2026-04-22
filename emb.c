/**
 * emb.c - Vector Embedding Library CLI
 * Summary: Command line interface for generating vector embeddings.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#include "emb.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Print command usage information.
 * @param name Program executable name.
 * @return None.
 */
static void kc_print_help(const char *name) {
    printf("Usage: %s <input> [options]\n", name);
    printf("\n");
    printf("Options:\n");
    printf("    -h, --help          Show this help message\n");
}

/**
 * Execute the command line interface.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Process status code.
 */
int main(int argc, char **argv) {
    kc_emb_t *ctx = NULL;
    char *input_text = NULL;
    int allocated = 0;
    int status = 0;

    float *vec = NULL;
    int dim = 0;

    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        kc_print_help(argv[0]);
        return 0;
    }

    if (argc < 2) {
        size_t size = 1024;
        size_t len = 0;
        const size_t max_size = 1048576;

        input_text = (char *)malloc(size);
        if (!input_text) {
            fprintf(stderr, "emb: out of memory\n");
            return 1;
        }
        allocated = 1;

        int c;
        while ((c = getchar()) != EOF) {
            if (len + 1 >= size) {
                if (size >= max_size) {
                    fprintf(stderr, "emb: input too large\n");
                    goto failure;
                }
                size *= 2;
                char *new_buf = (char *)realloc(input_text, size);
                if (!new_buf) {
                    fprintf(stderr, "emb: out of memory\n");
                    goto failure;
                }
                input_text = new_buf;
            }
            input_text[len++] = (char)c;
        }

        if (ferror(stdin)) {
            fprintf(stderr, "emb: read error\n");
            goto failure;
        }

        input_text[len] = '\0';

        if (len == 0) {
            kc_print_help(argv[0]);
            status = 1;
            goto failure;
        }
    } else {
        size_t total = 0;
        for (int i = 1; i < argc; i++) {
            total += strlen(argv[i]);
        }

        if (total == 0) {
            kc_print_help(argv[0]);
            return 1;
        }

        total += (argc - 2);
        input_text = (char *)malloc(total + 1);
        if (!input_text) {
            fprintf(stderr, "emb: out of memory\n");
            return 1;
        }
        allocated = 1;

        char *ptr = input_text;
        for (int i = 1; i < argc; i++) {
            size_t slen = strlen(argv[i]);
            memcpy(ptr, argv[i], slen);
            ptr += slen;
            if (i != argc - 1) {
                *ptr++ = ' ';
            }
        }
        *ptr = '\0';
    }

    ctx = kc_emb_open();
    if (!ctx) {
        fprintf(stderr, "emb: initialization failed\n");
        goto failure;
    }

    vec = kc_emb_exec(ctx, input_text);
    if (!vec) {
        fprintf(stderr, "emb: execution failed\n");
        goto failure;
    }

    dim = kc_emb_dim(ctx);
    if (dim <= 0) {
        fprintf(stderr, "emb: invalid embedding dimension\n");
        goto failure;
    }
    for (int i = 0; i < dim; i++) {
        printf("%f%s", vec[i], (i == dim - 1) ? "" : " ");
    }
    printf("\n");

    goto cleanup;

failure:
    status = 1;

cleanup:
    if (ctx) kc_emb_close(ctx);
    if (allocated) free(input_text);
    return status;
}
