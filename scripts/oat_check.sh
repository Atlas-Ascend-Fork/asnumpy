#!/bin/sh
# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

set -u

OAT_MIN_VERSION="1.0.2"

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "ERROR: oat-check must be run inside a Git repository."
    exit 1
}

REPO_NAME=$(basename "$REPO_ROOT")
REPORT_DIR="$REPO_ROOT/oat_reports"

# Find a usable Python 3 interpreter.
if command -v python3 >/dev/null 2>&1; then
    PYTHON=$(command -v python3)
elif command -v python >/dev/null 2>&1; then
    PYTHON=$(command -v python)
else
    echo "ERROR: Python 3 is required for OAT compliance checking."
    exit 1
fi

# oat-py provides the `oat` Python module.
if ! "$PYTHON" -c "import oat" >/dev/null 2>&1; then
    echo "OAT is not installed."
    echo "Install it with:"
    echo "  $PYTHON -m pip install 'oat-py>=$OAT_MIN_VERSION'"
    exit 1
fi

# pre-commit passes the files being checked as positional arguments.
if [ "$#" -eq 0 ]; then
    echo "OAT: no files to check."
    exit 0
fi

FILE_LIST=""

for file in "$@"; do
    # Ignore files that no longer exist.
    [ -f "$file" ] || continue

    case "$file" in
        /*)
            ABS_FILE="$file"
            ;;
        *)
            ABS_FILE="$REPO_ROOT/$file"
            ;;
    esac

    if [ -z "$FILE_LIST" ]; then
        FILE_LIST="$ABS_FILE"
    else
        FILE_LIST="$FILE_LIST,$ABS_FILE"
    fi
done

if [ -z "$FILE_LIST" ]; then
    echo "OAT: no existing files to check."
    exit 0
fi

mkdir -p "$REPORT_DIR"

# Use a project OAT configuration when one is available.
OAT_CONFIG=""

if [ -f "$REPO_ROOT/OAT.xml" ]; then
    OAT_CONFIG="$REPO_ROOT/OAT.xml"
elif [ -f "$REPO_ROOT/scripts/OAT.xml" ]; then
    OAT_CONFIG="$REPO_ROOT/scripts/OAT.xml"
fi

echo "Running OAT compliance check..."

if [ -n "$OAT_CONFIG" ]; then
    "$PYTHON" -m oat \
        -mode s \
        -s "$REPO_ROOT" \
        -r "$REPORT_DIR" \
        -n "$REPO_NAME" \
        -w 1 \
        -f "$FILE_LIST" \
        -oatconfig "$OAT_CONFIG"
else
    "$PYTHON" -m oat \
        -mode s \
        -s "$REPO_ROOT" \
        -r "$REPORT_DIR" \
        -n "$REPO_NAME" \
        -w 1 \
        -f "$FILE_LIST"
fi

OAT_EXIT=$?

REPORT_FILE="$REPORT_DIR/PlainReport_${REPO_NAME}.txt"
RESULT_FILE="$REPORT_DIR/result.txt"

if [ ! -f "$REPORT_FILE" ]; then
    echo "ERROR: OAT did not produce the expected report:"
    echo "  $REPORT_FILE"
    exit "${OAT_EXIT:-1}"
fi

INVALID_FILE_COUNT=$(
    grep "Invalid File Type Total Count" "$REPORT_FILE" 2>/dev/null |
        tail -1 |
        grep -o '[0-9][0-9]*' |
        tail -1
)

LICENSE_HEADER_COUNT=$(
    grep "License Header Invalid Total Count" "$REPORT_FILE" 2>/dev/null |
        tail -1 |
        grep -o '[0-9][0-9]*' |
        tail -1
)

INVALID_FILE_COUNT=${INVALID_FILE_COUNT:-0}
LICENSE_HEADER_COUNT=${LICENSE_HEADER_COUNT:-0}

{
    echo "OAT Compliance Check"
    echo "Invalid File Type Total Count: $INVALID_FILE_COUNT"
    echo "License Header Invalid Total Count: $LICENSE_HEADER_COUNT"
} > "$RESULT_FILE"

cat "$RESULT_FILE"

if [ "$INVALID_FILE_COUNT" -gt 0 ] ||
   [ "$LICENSE_HEADER_COUNT" -gt 0 ]; then
    echo "OAT compliance check failed."
    echo "See: $REPORT_FILE"
    exit 1
fi

if [ "$OAT_EXIT" -ne 0 ]; then
    echo "OAT completed with non-blocking findings."
    echo "See full report: $REPORT_FILE"
fi

echo "OAT compliance check passed."
exit 0
