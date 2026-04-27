#!/bin/sh
# test.sh
# Summary: Validation suite for emb functionality.
# Author:  KaisarCode
# Website: https://kaisarcode.com
# License: https://www.gnu.org/licenses/gpl-3.0.html

# Prints one failure line.
# @param 1 Failure message.
# @return 1 on failure.
kc_test_fail() {
    printf '\033[31m[FAIL]\033[0m %s\n' "$1"
    return 1
}

# Prints one success line.
# @param 1 Success message.
# @return 0 on success.
kc_test_pass() {
    printf '\033[32m[PASS]\033[0m %s\n' "$1"
}

# Verifies the binary exists.
# @return 0 on success, 1 on failure.
kc_test_check_binary() {
    if [ ! -x "./bin/x86_64/linux/emb" ]; then
        return 1
    fi

    return 0
}

# Runs the full validation suite.
# @return 0 on success, 1 on failure.
kc_test_main() {
    failed=0

    kc_test_check_binary || exit 1

    BIN="./bin/x86_64/linux/emb"

    if ! "$BIN" "test-input" > /dev/null 2>&1; then
        kc_test_fail "basic execution (parameter)"
        failed=$((failed + 1))
    else
        kc_test_pass "basic execution (parameter)"
    fi

    if ! printf "test-input-stdin" | "$BIN" > /dev/null 2>&1; then
        kc_test_fail "basic execution (stdin)"
        failed=$((failed + 1))
    else
        kc_test_pass "basic execution (stdin)"
    fi

    if ! KC_EMB_THREADS=1 "$BIN" "test-input" > /dev/null 2>&1; then
        kc_test_fail "KC_EMB_THREADS valid (1)"
        failed=$((failed + 1))
    else
        kc_test_pass "KC_EMB_THREADS valid (1)"
    fi

    for val in "0" "-1" "+1" "4x" "abc"; do
        if KC_EMB_THREADS="$val" "$BIN" "test-input" > /dev/null 2>&1; then
            kc_test_fail "KC_EMB_THREADS invalid ($val)"
            failed=$((failed + 1))
        else
            kc_test_pass "KC_EMB_THREADS invalid ($val)"
        fi
    done

    if [ "$failed" -eq 0 ]; then
        return 0
    fi

    return 1
}

kc_test_main
