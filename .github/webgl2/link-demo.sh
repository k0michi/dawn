#!/bin/sh

set -eu

# Links demo.cpp against the build from build.sh into <output>/index.html.
#
# Usage: .github/webgl2/link-demo.sh <build directory> <output directory>

webgl2=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source=$(CDPATH= cd -- "$webgl2/../.." && pwd)
build=$1
output=$2

mkdir -p "$output"
em++ -std=c++20 -O2 -g2 -DDAWN_EMSCRIPTEN_NATIVE \
    -I"$source/include" -I"$build/gen/include" \
    "$webgl2/demo.cpp" "$build/src/dawn/native/libwebgpu_dawn.a" \
    -sMIN_WEBGL_VERSION=2 -sMAX_WEBGL_VERSION=2 -sGL_ENABLE_GET_PROC_ADDRESS \
    -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=1MB -sASSERTIONS \
    --shell-file "$webgl2/shell.html" -o "$output/index.html"
