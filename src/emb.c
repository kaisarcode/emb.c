/**
 * emb.c - Vector Embedding Library CLI
 * Summary: Command line interface for generating vector embeddings.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "emb.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KC_EMB_VERSION "0.1.0"

#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#define isatty _isatty
#define STDIN_FILENO 0
#endif

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
    printf("    -v, --version       Show version\n");
}

/**
 * Print command version information.
 * @return None.
 */
static void kc_print_version(void) {
    printf("emb %s\n", KC_EMB_VERSION);
}

/**
 * Read one complete line from a stream.
 * @param stream Input stream.
 * @return Allocated line without trailing newline, or NULL on EOF/error.
 */
static char *kc_read_line(FILE *stream) {
    char *buf = NULL;
    size_t len = 0;
    size_t cap = 0;
    int ch;

    while ((ch = fgetc(stream)) != EOF) {
        char *next;

        if (ch == '\n') {
            break;
        }

        if (len + 1 >= cap) {
            cap = cap ? cap * 2 : 256;
            next = (char *)realloc(buf, cap);

            if (!next) {
                free(buf);
                return NULL;
            }

            buf = next;
        }

        buf[len++] = (char)ch;
    }

    if (ch == EOF && len == 0) {
        free(buf);
        return NULL;
    }

    if (len + 1 >= cap) {
        char *next;
        cap = cap ? cap + 1 : 1;
        next = (char *)realloc(buf, cap);

        if (!next) {
            free(buf);
            return NULL;
        }

        buf = next;
    }

    buf[len] = '\0';
    return buf;
}

/**
 * Generate and print one embedding.
 * @param ctx   Embedding context.
 * @param input Input text.
 * @param dim   Embedding dimension.
 * @param vec   Output vector buffer.
 * @return 0 on success, -1 on failure.
 */
static int kc_run_emb(kc_emb_t *ctx, const char *input, int dim, float *vec) {
    if (!input || *input == '\0') {
        return 0;
    }

    if (kc_emb_exec(ctx, input, vec) != KC_EMB_OK) {
        return -1;
    }

    for (int i = 0; i < dim; i++) {
        printf("%.6f%c", vec[i], (i == dim - 1) ? '\n' : ' ');
    }

    fflush(stdout);
    return 0;
}

/**
 * Execute the command line interface.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Process status code.
 */
int main(int argc, char **argv) {
    kc_emb_t *ctx = NULL;
    int status = 0;
    float *vec = NULL;
    int dim = 0;

    if (argc >= 2) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            kc_print_help(argv[0]);
            return 0;
        }
        if (strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0) {
            kc_print_version();
            return 0;
        }
    }

    ctx = kc_emb_open();
    if (!ctx) {
        fprintf(stderr, "emb: initialization failed\n");
        return 1;
    }

    dim = kc_emb_dim(ctx);
    if (dim <= 0) {
        fprintf(stderr, "emb: invalid embedding dimension\n");
        kc_emb_close(ctx);
        return 1;
    }

    vec = (float *)malloc((size_t)dim * sizeof(float));
    if (!vec) {
        fprintf(stderr, "emb: out of memory\n");
        kc_emb_close(ctx);
        return 1;
    }

    if (argc >= 2) {
        size_t total = 0;
        for (int i = 1; i < argc; i++) {
            total += strlen(argv[i]);
        }

        total += (argc - 2);
        char *input_text = (char *)malloc(total + 1);
        if (!input_text) {
            fprintf(stderr, "emb: out of memory\n");
            status = 1;
            goto cleanup;
        }

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

        if (kc_run_emb(ctx, input_text, dim, vec) != 0) {
            fprintf(stderr, "emb: execution failed\n");
            status = 1;
        }
        free(input_text);
    } else {
        if (isatty(STDIN_FILENO)) {
            kc_print_help(argv[0]);
            status = 1;
            goto cleanup;
        }

        for (;;) {
            char *line = kc_read_line(stdin);
            if (!line) {
                break;
            }

            if (kc_run_emb(ctx, line, dim, vec) != 0) {
                fprintf(stderr, "emb: execution failed\n");
                status = 1;
            }

            free(line);
        }
    }

cleanup:
    if (vec) free(vec);
    if (ctx) kc_emb_close(ctx);
    return status;
}
