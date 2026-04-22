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
    printf '[FAIL] %s\n' "$1"
    return 1
}

# Prints one success line.
# @param 1 Success message.
# @return 0 on success.
kc_test_pass() {
    printf '[PASS] %s\n' "$1"
}

# Verifies the binary exists.
# @return 0 on success, 1 on failure.
kc_test_check_binary() {
    if [ ! -x "./emb" ]; then
        printf '%s\n' '[ERROR] emb binary not found. Please compile first.'
        return 1
    fi

    return 0
}

# Runs the full validation suite.
# @return 0 on success, 1 on failure.
kc_test_main() {
    failed=0

    kc_test_check_binary || exit 1

    if ! ./emb "test-input" > /dev/null; then
        kc_test_fail "basic execution"
        failed=$((failed + 1))
    else
        kc_test_pass "basic execution"
    fi

    if [ "$failed" -eq 0 ]; then
        printf '%s\n' '[SUCCESS] All tests passed!'
        return 0
    fi

    printf '[FAILURE] %s tests failed.\n' "$failed"
    return 1
}

kc_test_main
