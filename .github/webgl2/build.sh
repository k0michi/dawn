#!/bin/sh

set -eu

# Builds dawn::native for Emscripten with the OpenGL ES backend on WebGL 2, as
# the static library <build>/src/dawn/native/libwebgpu_dawn.a.
#
# Usage: .github/webgl2/build.sh <build directory>
#
# The dependencies must have been fetched with tools/fetch_dawn_dependencies.py.
# Set EM_COMPILER_WRAPPER=ccache to cache compiles. EM_CACHE, if set, must not
# go through a symlink such as /tmp on macOS.

source=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
build=$1

emcmake cmake -G Ninja -S "$source" -B "$build" \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DDAWN_EMSCRIPTEN_NATIVE=ON \
    -DDAWN_BUILD_MONOLITHIC_LIBRARY=STATIC \
    -DBUILD_SHARED_LIBS=OFF \
    -DDAWN_BUILD_PROTOBUF=OFF \
    -DDAWN_BUILD_SAMPLES=OFF \
    -DDAWN_BUILD_TESTS=OFF \
    -DDAWN_USE_GLFW=OFF \
    -DDAWN_ENABLE_NULL=OFF \
    -DDAWN_ENABLE_OPENGLES=ON \
    -DDAWN_ENABLE_DESKTOP_GL=OFF \
    -DDAWN_ENABLE_VULKAN=OFF \
    -DDAWN_ENABLE_SPIRV_VALIDATION=OFF \
    -DDAWN_USE_X11=OFF \
    -DDAWN_USE_WAYLAND=OFF \
    -DTINT_BUILD_CMD_TOOLS=OFF \
    -DTINT_BUILD_TESTS=OFF \
    -DTINT_BUILD_SPV_READER=OFF \
    -DTINT_BUILD_SPV_WRITER=OFF \
    -DTINT_BUILD_HLSL_WRITER=OFF \
    -DTINT_BUILD_MSL_WRITER=OFF \
    -DTINT_BUILD_GLSL_WRITER=ON \
    -DTINT_BUILD_GLSL_VALIDATOR=OFF \
    -DTINT_BUILD_NULL_WRITER=ON \
    -DTINT_BUILD_IR_BINARY=OFF
ninja -C "$build" webgpu_dawn
