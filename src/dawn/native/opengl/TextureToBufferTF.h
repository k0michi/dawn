// Copyright 2026 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef SRC_DAWN_NATIVE_OPENGL_TEXTURETOBUFFERTF_H_
#define SRC_DAWN_NATIVE_OPENGL_TEXTURETOBUFFERTF_H_

#include "absl/container/flat_hash_map.h"
#include "src/dawn/native/Commands.h"
#include "src/dawn/native/Error.h"
#include "src/dawn/native/opengl/opengl_platform.h"

namespace dawn::native::opengl {

struct OpenGLFunctions;
class Device;
class Texture;

// Texture-to-buffer copies of the formats that glReadPixels can't read on WebGL 2, where Dawn
// elsewhere copies with a compute shader. WebGL 2 has no compute shaders: instead, a vertex shader
// with rasterization discarded fetches texels and packs them into words, which transform feedback
// writes to a staging buffer, and the rows are then copied to their places in the destination.

// Whether a copy from |texture|'s |aspect| uses CopyTextureToBufferWithTransformFeedback.
bool UsesTransformFeedbackForTextureToBufferCopy(const Texture* texture, Aspect aspect);

MaybeError CopyTextureToBufferWithTransformFeedback(const OpenGLFunctions& gl,
                                                    Device* device,
                                                    const TextureCopy& source,
                                                    const BufferCopy& destination,
                                                    const Extent3D& copySize);

// WebGL 2 without WEBGL_stencil_texturing can't sample stencil values. Before a render pass samples
// the stencil of |texture|, writes its stencil values in |range| to the texture's stencil shadow,
// an R8UI texture of the same shape, which bindings then use in its place.
MaybeError UpdateStencilShadow(const OpenGLFunctions& gl,
                               Device* device,
                               Texture* texture,
                               const SubresourceRange& range);

// The GL objects the copies and other WebGL 2 emulations use, kept for the lifetime of the
// device's context.
struct TextureToBufferTFCache {
    absl::flat_hash_map<uint32_t, GLuint> programs;
    GLuint stencilBitsProgram = 0;
    GLuint stencilShadowProgram = 0;
    GLuint sampler = 0;
    GLuint vertexArray = 0;
    // uint32 indices for uint16 list draws with index 0xFFFF (see DrawIndexed in
    // CommandBufferGL.cpp). Only ever bound to GL_ELEMENT_ARRAY_BUFFER, as WebGL requires.
    GLuint wideIndexBuffer = 0;
};

}  // namespace dawn::native::opengl

#endif  // SRC_DAWN_NATIVE_OPENGL_TEXTURETOBUFFERTF_H_
