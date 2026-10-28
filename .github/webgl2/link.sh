#!/bin/sh

set -eu

# Links <program>.cpp from this directory against the build from build.sh into
# <output>/index.html.
#
# Usage: .github/webgl2/link.sh <program> <build directory> <output directory>
#
# The programs are demo (the swift-game textured quad, for smoke-test.mjs),
# capability-test, buffer-test and async-test (for run-tests.mjs), and
# surface-test (for surface-test.mjs).

webgl2=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source=$(CDPATH= cd -- "$webgl2/../.." && pwd)
program=$1
build=$2
output=$3

mkdir -p "$output"
em++ -std=c++20 -O2 -g2 -DDAWN_EMSCRIPTEN_NATIVE \
    -I"$source/include" -I"$build/gen/include" \
    "$webgl2/$program.cpp" "$build/src/dawn/native/libwebgpu_dawn.a" \
    -sMIN_WEBGL_VERSION=2 -sMAX_WEBGL_VERSION=2 -sGL_ENABLE_GET_PROC_ADDRESS \
    -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=1MB -sASSERTIONS \
    --shell-file "$webgl2/shell.html" -o "$output/index.html"
