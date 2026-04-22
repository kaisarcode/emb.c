# Agent Protocol: KaisarCode C Library Blueprint

## Overview
This directory is the master library for all C libraries developed by KaisarCode. AI agents MUST follow these patterns when creating, refactoring, or expanding any library based on this template.

## Architectural Patterns
1. **Flat Structure**: Do not create `src/` or `include/` directories. All `.c`, `.h`, and `.sh` files reside in the root.
2. **File Naming**:
    *   `<name>.c`: Command-line interface and `main()` function.
    *   `lib<name>.c`: Core library implementation.
    *   `<name>.h`: Public API header.
    *   `test.sh`: Shell-based validation suite.
3. **Naming Directory**: The project directory MUST end with a `.c` extension (e.g., `my-lib.c/`).

## Mandatory Coding Standards (KCS)
Refer to `KCS.md` in the workspace root for the full ruleset. Key requirements:
- **No Internal Comments**: Comments inside functions are strictly forbidden. Use self-descriptive code or DocBlocks.
- **Strict DocBlocks**: Every function, struct, and file must have a `/** ... */` DocBlock.
- **Tag Requirements**: DocBlocks must include `@param` and `@return` tags.
- **Line Limits**: DocBlock content is limited to 80 characters per line.
- **Indentation**: Use exactly 4-space increments.
- **Prefixing**: All exported symbols and internal static functions must use the `kc_` prefix.
- **POSIX First**: Implementation files must define `_POSIX_C_SOURCE` and `_XOPEN_SOURCE` before any includes.

## Agent Workflow for New Libraries
1. **Copy**: `cp -r emb.c <new-name>.c`
2. **Rename**: Update filenames and replace all occurrences of `emb` with `<new-name>`.
3. **Symbol Update**: Ensure symbols follow `kc_<new-name>_` for public APIs.
4. **Validation**: Proactively run the KCS validator (e.g., `kc-kcs`) against the new project to ensure 100% compliance.

---

**Author:** KaisarCode

**Email:** <kaisar@kaisarcode.com>

**Website:** [https://kaisarcode.com](https://kaisarcode.com)

**License:** [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

© 2026 KaisarCode
