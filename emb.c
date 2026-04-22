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
    kc_emb_t *ctx;
    int rc;
    char *input_text = NULL;

    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        kc_print_help(argv[0]);
        return 0;
    }

    if (argc < 2) {
        size_t size = 1024;
        size_t len = 0;
        input_text = (char *)malloc(size);
        if (!input_text) {
            fprintf(stderr, "emb: out of memory\n");
            return 1;
        }
        int c;
        while ((c = getchar()) != EOF) {
            if (len + 1 >= size) {
                size *= 2;
                char *new_buf = (char *)realloc(input_text, size);
                if (!new_buf) {
                    free(input_text);
                    fprintf(stderr, "emb: out of memory\n");
                    return 1;
                }
                input_text = new_buf;
            }
            input_text[len++] = (char)c;
        }
        input_text[len] = '\0';

        if (len == 0) {
            kc_print_help(argv[0]);
            free(input_text);
            return 1;
        }
    } else {
        input_text = argv[1];
    }

    ctx = kc_emb_open();
    if (!ctx) {
        fprintf(stderr, "emb: out of memory\n");
        if (argc < 2) free(input_text);
        return 1;
    }

    rc = kc_emb_exec(ctx, input_text);

    if (rc != KC_EMB_OK) {
        fprintf(stderr, "emb: execution failed\n");
        kc_emb_close(ctx);
        if (argc < 2) free(input_text);
        return 1;
    }

    kc_emb_close(ctx);
    if (argc < 2) free(input_text);
    return 0;
}
