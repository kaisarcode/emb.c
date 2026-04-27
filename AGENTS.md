# Agent Protocol: emb.c

## 1. Structure
src/emb.c — CLI/main only
src/libemb.c — implementation only
src/emb.h — public API only
lib/ggml/ — vendored GGML (non-KC, skip KCS validation)
lib/model.gguf — embedded model binary (Git LFS)
test.sh — external validation
bin/{arch}/{platform}/ — artifacts, committed, Git LFS
.build/ — temp, gitignored

artifacts-per-target: executable + libemb.a + libemb.so (or .dll/.dll.a on windows)
model-embedding: objcopy converts lib/model.gguf to model.o per target at build time
same-layout-for-libs-and-apps: no exceptions

## 2. Rules
api:no-invent — README header CLI tests impl must match. placeholders marked explicitly.
build:no-march-native-default — native opt behind EMB_NATIVE=OFF
threading:single-context — serialized by caller, KC_EMB_THREADS controls CPU threads
lifecycle:document-ownership — who owns pointer, how long valid, safe to close while active?
scope:no-redesign — no frameworks hidden services background workers global state unless asked. return modified files only.

## 3. Validation
before-returning: README matches CLI+API+build+threading. tests cover failure cases. no inline comments inside functions.
