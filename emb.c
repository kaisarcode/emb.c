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

    if (argc < 2) {
        kc_print_help(argv[0]);
        return 1;
    }

    if (
        strcmp(argv[1], "-h") == 0 ||
        strcmp(argv[1], "--help") == 0
    ) {
        kc_print_help(argv[0]);
        return 0;
    }

    ctx = kc_emb_open();
    if (!ctx) {
        fprintf(stderr, "emb: out of memory\n");
        return 1;
    }

    rc = kc_emb_exec(ctx, argv[1]);

    if (rc != KC_EMB_OK) {
        fprintf(stderr, "emb: execution failed\n");
        kc_emb_close(ctx);
        return 1;
    }

    kc_emb_close(ctx);
    return 0;
}
