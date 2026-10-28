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

#include "src/dawn/native/opengl/TextureToBufferTF.h"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "src/dawn/native/BlockInfo.h"
#include "src/dawn/native/opengl/BufferGL.h"
#include "src/dawn/native/opengl/DeviceGL.h"
#include "src/dawn/native/opengl/TextureGL.h"
#include "src/dawn/native/opengl/UtilsGL.h"
#include "src/utils/numeric.h"

#if defined(__EMSCRIPTEN__)
// Emscripten's WebGL 2 getBufferSubData, which its GetProcAddress doesn't return.
extern "C" void glGetBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, void* data);
#endif

namespace dawn::native::opengl {

namespace {

// How a texel's channels are packed into bytes.
enum class Packing : uint32_t {
    Unorm8,
    BGRA8Unorm,
    Uint8,
    Sint8,
    Uint16,
    Sint16,
    Uint32,
    Sint32,
    RGB10A2Uint,
    Float16,
    Float32,
};

struct Layout {
    Packing packing;
    uint32_t bytesPerTexel;
};

std::optional<Layout> GetLayout(wgpu::TextureFormat format) {
    switch (format) {
        case wgpu::TextureFormat::R8Unorm:
            return Layout{Packing::Unorm8, 1};
        case wgpu::TextureFormat::RG8Unorm:
            return Layout{Packing::Unorm8, 2};
        case wgpu::TextureFormat::BGRA8Unorm:
            return Layout{Packing::BGRA8Unorm, 4};
        case wgpu::TextureFormat::R8Uint:
            return Layout{Packing::Uint8, 1};
        case wgpu::TextureFormat::RG8Uint:
            return Layout{Packing::Uint8, 2};
        case wgpu::TextureFormat::RGBA8Uint:
            return Layout{Packing::Uint8, 4};
        case wgpu::TextureFormat::R8Sint:
            return Layout{Packing::Sint8, 1};
        case wgpu::TextureFormat::RG8Sint:
            return Layout{Packing::Sint8, 2};
        case wgpu::TextureFormat::RGBA8Sint:
            return Layout{Packing::Sint8, 4};
        case wgpu::TextureFormat::R16Uint:
            return Layout{Packing::Uint16, 2};
        case wgpu::TextureFormat::RG16Uint:
            return Layout{Packing::Uint16, 4};
        case wgpu::TextureFormat::RGBA16Uint:
            return Layout{Packing::Uint16, 8};
        case wgpu::TextureFormat::R16Sint:
            return Layout{Packing::Sint16, 2};
        case wgpu::TextureFormat::RG16Sint:
            return Layout{Packing::Sint16, 4};
        case wgpu::TextureFormat::RGBA16Sint:
            return Layout{Packing::Sint16, 8};
        case wgpu::TextureFormat::R32Uint:
            return Layout{Packing::Uint32, 4};
        case wgpu::TextureFormat::RG32Uint:
            return Layout{Packing::Uint32, 8};
        case wgpu::TextureFormat::R32Sint:
            return Layout{Packing::Sint32, 4};
        case wgpu::TextureFormat::RG32Sint:
            return Layout{Packing::Sint32, 8};
        case wgpu::TextureFormat::RGB10A2Uint:
            return Layout{Packing::RGB10A2Uint, 4};
        case wgpu::TextureFormat::R16Float:
            return Layout{Packing::Float16, 2};
        case wgpu::TextureFormat::RG16Float:
            return Layout{Packing::Float16, 4};
        case wgpu::TextureFormat::RGBA16Float:
            return Layout{Packing::Float16, 8};
        case wgpu::TextureFormat::R32Float:
            return Layout{Packing::Float32, 4};
        case wgpu::TextureFormat::RG32Float:
            return Layout{Packing::Float32, 8};
        default:
            return std::nullopt;
    }
}

// The GLSL ES 3.00 sampler type for |target|, without the u or i prefix. texelFetch() can't
// read cube maps in ES 3.00.
const char* SamplerType(GLenum target) {
    switch (target) {
        case GL_TEXTURE_2D:
            return "sampler2D";
        case GL_TEXTURE_2D_ARRAY:
            return "sampler2DArray";
        case GL_TEXTURE_3D:
            return "sampler3D";
        default:
            return nullptr;
    }
}

// A vertex shader in which vertex i packs bytes 4i to 4i+3 of the copied texels, laid out
// without padding, into `word`, which transform feedback writes to the staging buffer.
std::string VertexShader(Packing packing, uint32_t bytesPerTexel, GLenum target) {
    std::string prefix;
    std::string value;
    switch (packing) {
        case Packing::Uint8:
        case Packing::Uint16:
        case Packing::Uint32:
        case Packing::RGB10A2Uint:
            prefix = "u";
            value = "uvec4";
            break;
        case Packing::Sint8:
        case Packing::Sint16:
        case Packing::Sint32:
            prefix = "i";
            value = "ivec4";
            break;
        default:
            value = "vec4";
            break;
    }
    std::string sampler = prefix + SamplerType(target);

    // The texel's bytes as little-endian words, from its channels in `v`.
    std::string pack;
    switch (packing) {
        case Packing::Unorm8:
        case Packing::BGRA8Unorm:
            pack = "uvec4 c = uvec4(round(clamp(v, 0.0, 1.0) * 255.0));\n";
            if (packing == Packing::BGRA8Unorm) {
                pack += "c = c.zyxw;\n";
            }
            pack += "return uvec4(c.x | (c.y << 8u) | (c.z << 16u) | (c.w << 24u), 0u, 0u, 0u);\n";
            break;
        case Packing::Uint8:
        case Packing::Sint8:
            pack =
                "uvec4 c = uvec4(v) & 0xffu;\n"
                "return uvec4(c.x | (c.y << 8u) | (c.z << 16u) | (c.w << 24u), 0u, 0u, 0u);\n";
            break;
        case Packing::Uint16:
        case Packing::Sint16:
            pack =
                "uvec4 c = uvec4(v) & 0xffffu;\n"
                "return uvec4(c.x | (c.y << 16u), c.z | (c.w << 16u), 0u, 0u);\n";
            break;
        case Packing::Uint32:
        case Packing::Sint32:
            pack = "return uvec4(v);\n";
            break;
        case Packing::RGB10A2Uint:
            pack = "return uvec4(v.r | (v.g << 10u) | (v.b << 20u) | (v.a << 30u), 0u, 0u, 0u);\n";
            break;
        case Packing::Float16:
            pack = "return uvec4(packHalf2x16(v.xy), packHalf2x16(v.zw), 0u, 0u);\n";
            break;
        case Packing::Float32:
            pack = "return floatBitsToUint(v);\n";
            break;
    }

    std::string fetch = target == GL_TEXTURE_2D
                            ? "texelFetch(src, origin.xy + ivec2(int(x), int(y)), 0)"
                            : "texelFetch(src, origin + ivec3(int(x), int(y), int(z)), 0)";

    return "#version 300 es\n"
           "precision highp float;\n"
           "precision highp int;\n"
           "uniform highp " +
           sampler +
           " src;\n"
           "uniform ivec3 origin;\n"
           "uniform uvec3 copySize;\n"
           "flat out uint word;\n"
           "uvec4 TexelWords(uint x, uint y, uint z) {\n" +
           value + " v = " + fetch + ";\n" + pack +
           "}\n"
           "void main() {\n"
           "    const uint kBytesPerTexel = " +
           std::to_string(bytesPerTexel) +
           "u;\n"
           "    uint texelCount = copySize.x * copySize.y * copySize.z;\n"
           "    uint result = 0u;\n"
           "    for (uint b = 0u; b < 4u; b++) {\n"
           "        uint byteIndex = uint(gl_VertexID) * 4u + b;\n"
           "        uint texel = byteIndex / kBytesPerTexel;\n"
           "        if (texel >= texelCount) {\n"
           "            break;\n"
           "        }\n"
           "        uint inTexel = byteIndex % kBytesPerTexel;\n"
           "        uvec4 words = TexelWords(texel % copySize.x, (texel / copySize.x) % copySize.y,\n"
           "                                 texel / (copySize.x * copySize.y));\n"
           "        result |= ((words[inTexel / 4u] >> (8u * (inTexel % 4u))) & 0xffu) << (8u * b);\n"
           "    }\n"
           "    word = result;\n"
           "}\n";
}

// GLSL ES 3.00 programs need a fragment shader even when rasterization is discarded.
constexpr const char* kFragmentShader =
    "#version 300 es\n"
    "void main() {}\n";

ResultOrError<GLuint> CompileShader(const OpenGLFunctions& gl,
                                    GLenum type,
                                    const std::string& source) {
    GLuint shader = DAWN_GL_TRY(gl, CreateShader(type));
    const char* text = source.c_str();
    DAWN_GL_TRY(gl, ShaderSource(shader, 1, &text, nullptr));
    DAWN_GL_TRY(gl, CompileShader(shader));
    GLint status = GL_FALSE;
    DAWN_GL_TRY(gl, GetShaderiv(shader, GL_COMPILE_STATUS, &status));
    if (status == GL_FALSE) {
        GLint length = 0;
        DAWN_GL_TRY(gl, GetShaderiv(shader, GL_INFO_LOG_LENGTH, &length));
        std::vector<char> log(std::max(length, 1));
        DAWN_GL_TRY(gl, GetShaderInfoLog(shader, length, nullptr, log.data()));
        DAWN_GL_TRY(gl, DeleteShader(shader));
        return DAWN_UNRECOVERABLE_ERROR(
            "Compiling the shader for a texture-to-buffer copy failed:\n%s\n%s", text, log.data());
    }
    return shader;
}

ResultOrError<GLuint> GetProgram(const OpenGLFunctions& gl,
                                 TextureToBufferTFCache& cache,
                                 const Layout& layout,
                                 GLenum target) {
    uint32_t key = (static_cast<uint32_t>(layout.packing) << 16) | (layout.bytesPerTexel << 8) |
                   (target == GL_TEXTURE_2D ? 0 : target == GL_TEXTURE_2D_ARRAY ? 1 : 2);
    if (auto it = cache.programs.find(key); it != cache.programs.end()) {
        return it->second;
    }

    GLuint vertexShader;
    DAWN_TRY_ASSIGN(vertexShader,
                    CompileShader(gl, GL_VERTEX_SHADER,
                                  VertexShader(layout.packing, layout.bytesPerTexel, target)));
    GLuint fragmentShader;
    DAWN_TRY_ASSIGN(fragmentShader, CompileShader(gl, GL_FRAGMENT_SHADER, kFragmentShader));

    GLuint program = DAWN_GL_TRY(gl, CreateProgram());
    DAWN_GL_TRY(gl, AttachShader(program, vertexShader));
    DAWN_GL_TRY(gl, AttachShader(program, fragmentShader));
    const char* varying = "word";
    DAWN_GL_TRY(gl, TransformFeedbackVaryings(program, 1, &varying, GL_INTERLEAVED_ATTRIBS));
    DAWN_GL_TRY(gl, LinkProgram(program));
    DAWN_GL_TRY(gl, DeleteShader(vertexShader));
    DAWN_GL_TRY(gl, DeleteShader(fragmentShader));
    GLint status = GL_FALSE;
    DAWN_GL_TRY(gl, GetProgramiv(program, GL_LINK_STATUS, &status));
    if (status == GL_FALSE) {
        GLint length = 0;
        DAWN_GL_TRY(gl, GetProgramiv(program, GL_INFO_LOG_LENGTH, &length));
        std::vector<char> log(std::max(length, 1));
        DAWN_GL_TRY(gl, GetProgramInfoLog(program, length, nullptr, log.data()));
        DAWN_GL_TRY(gl, DeleteProgram(program));
        return DAWN_UNRECOVERABLE_ERROR("Linking the program for a texture-to-buffer copy failed:\n%s",
                                   log.data());
    }
    cache.programs.emplace(key, program);
    return program;
}

}  // namespace

bool UsesTransformFeedbackForTextureToBufferCopy(const Texture* texture, Aspect aspect) {
#if defined(__EMSCRIPTEN__)
    return aspect == Aspect::Color && GetLayout(texture->GetFormat().format).has_value() &&
           SamplerType(texture->GetGLTarget()) != nullptr;
#else
    return false;
#endif
}

MaybeError CopyTextureToBufferWithTransformFeedback(const OpenGLFunctions& gl,
                                                    Device* device,
                                                    const TextureCopy& source,
                                                    const BufferCopy& destination,
                                                    const Extent3D& copySize) {
#if !defined(__EMSCRIPTEN__)
    DAWN_UNREACHABLE();
#else
    Texture* texture = ToBackend(source.texture.Get());
    Buffer* buffer = ToBackend(destination.buffer.Get());
    Layout layout = *GetLayout(texture->GetFormat().format);
    GLenum target = texture->GetGLTarget();
    TextureToBufferTFCache& cache = device->GetTextureToBufferTFCache();

    GLuint program;
    DAWN_TRY_ASSIGN(program, GetProgram(gl, cache, layout, target));
    if (cache.sampler == 0) {
        // Integer textures are incomplete with linear filtering, even for texelFetch().
        DAWN_GL_TRY(gl, GenSamplers(1, &cache.sampler));
        DAWN_GL_TRY(gl, SamplerParameteri(cache.sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        DAWN_GL_TRY(gl, SamplerParameteri(cache.sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        // A vertex array with no attributes, so that no vertex buffer limits the vertex count.
        DAWN_GL_TRY(gl, GenVertexArrays(1, &cache.vertexArray));
    }

    // The copied bytes without row or image padding, rounded up to whole words.
    const uint64_t rowBytes = uint64_t(copySize.width) * layout.bytesPerTexel;
    const uint64_t copyBytes = rowBytes * copySize.height * copySize.depthOrArrayLayers;
    const GLsizei wordCount = checked_cast<GLsizei>((copyBytes + 3) / 4);

    DAWN_GL_TRY(gl, UseProgram(program));
    DAWN_GL_TRY(gl, ActiveTexture(GL_TEXTURE0));
    DAWN_GL_TRY(gl, BindTexture(target, texture->GetTextureHandle()));
    // texelFetch() levels count from the base level. Binding a view sets both levels again.
    DAWN_GL_TRY(gl, TexParameteri(target, GL_TEXTURE_BASE_LEVEL, source.mipLevel));
    DAWN_GL_TRY(gl, TexParameteri(target, GL_TEXTURE_MAX_LEVEL, source.mipLevel));
    DAWN_GL_TRY(gl, BindSampler(0, cache.sampler));
    GLint location = DAWN_GL_TRY(gl, GetUniformLocation(program, "src"));
    DAWN_GL_TRY(gl, Uniform1i(location, 0));
    location = DAWN_GL_TRY(gl, GetUniformLocation(program, "origin"));
    DAWN_GL_TRY(gl, Uniform3i(location, dchecked_cast<GLint>(source.origin.x),
                              dchecked_cast<GLint>(source.origin.y),
                              dchecked_cast<GLint>(source.origin.z)));
    location = DAWN_GL_TRY(gl, GetUniformLocation(program, "copySize"));
    DAWN_GL_TRY(gl, Uniform3ui(location, copySize.width, copySize.height,
                               copySize.depthOrArrayLayers));
    DAWN_GL_TRY(gl, BindVertexArray(cache.vertexArray));

    GLuint staging = 0;
    DAWN_GL_TRY(gl, GenBuffers(1, &staging));
    DAWN_GL_TRY(gl, BindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, staging));
    DAWN_GL_TRY(gl, BufferData(GL_TRANSFORM_FEEDBACK_BUFFER, GLsizeiptr(wordCount) * 4, nullptr,
                               GL_STREAM_COPY));
    DAWN_GL_TRY(gl, BindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, staging));
    DAWN_GL_TRY(gl, Enable(GL_RASTERIZER_DISCARD));
    DAWN_GL_TRY(gl, BeginTransformFeedback(GL_POINTS));
    DAWN_GL_TRY(gl, DrawArrays(GL_POINTS, 0, wordCount));
    DAWN_GL_TRY(gl, EndTransformFeedback());
    DAWN_GL_TRY(gl, Disable(GL_RASTERIZER_DISCARD));
    // WebGL doesn't let a buffer bound for transform feedback be bound to other targets.
    DAWN_GL_TRY(gl, BindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0));
    DAWN_GL_TRY(gl, BindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0));
    DAWN_GL_TRY(gl, BindVertexArray(0));
    DAWN_GL_TRY(gl, BindSampler(0, 0));

    // Copy each row to its place, leaving the padding between rows and images as it was.
    const TypedTexelBlockInfo& blockInfo = GetBlockInfo(source);
    const uint64_t bytesPerRow = blockInfo.ToBytes(destination.blocksPerRow);
    const uint64_t bytesPerImage =
        blockInfo.ToBytes(destination.blocksPerRow * destination.rowsPerImage);
    auto ForEachRow = [&](auto copyRow) -> MaybeError {
        for (uint32_t z = 0; z < copySize.depthOrArrayLayers; z++) {
            for (uint32_t y = 0; y < copySize.height; y++) {
                DAWN_TRY(copyRow((uint64_t(z) * copySize.height + y) * rowBytes,
                                 destination.offset + z * bytesPerImage + y * bytesPerRow));
            }
        }
        return {};
    };
    if (buffer->GetDataTarget() == GL_ELEMENT_ARRAY_BUFFER) {
        // WebGL can't copy between an index buffer and another buffer on the GPU.
        std::vector<uint8_t> data(copyBytes);
        DAWN_GL_TRY(gl, BindBuffer(GL_COPY_READ_BUFFER, staging));
        glGetBufferSubData(GL_COPY_READ_BUFFER, 0, checked_cast<GLsizeiptr>(copyBytes),
                           data.data());
        DAWN_GL_TRY(gl, BindBuffer(GL_COPY_READ_BUFFER, 0));
        DAWN_GL_TRY(gl, BindBuffer(GL_COPY_WRITE_BUFFER, buffer->GetHandle()));
        DAWN_TRY(ForEachRow([&](uint64_t from, uint64_t to) -> MaybeError {
            DAWN_GL_TRY(gl, BufferSubData(GL_COPY_WRITE_BUFFER, checked_cast<GLintptr>(to),
                                          checked_cast<GLsizeiptr>(rowBytes), data.data() + from));
            return {};
        }));
        DAWN_GL_TRY(gl, BindBuffer(GL_COPY_WRITE_BUFFER, 0));
    } else {
        DAWN_GL_TRY(gl, BindBuffer(GL_COPY_READ_BUFFER, staging));
        DAWN_GL_TRY(gl, BindBuffer(GL_COPY_WRITE_BUFFER, buffer->GetHandle()));
        DAWN_TRY(ForEachRow([&](uint64_t from, uint64_t to) -> MaybeError {
            DAWN_GL_TRY(gl, CopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                                              checked_cast<GLintptr>(from),
                                              checked_cast<GLintptr>(to),
                                              checked_cast<GLsizeiptr>(rowBytes)));
            return {};
        }));
        DAWN_GL_TRY(gl, BindBuffer(GL_COPY_READ_BUFFER, 0));
        DAWN_GL_TRY(gl, BindBuffer(GL_COPY_WRITE_BUFFER, 0));
    }
    DAWN_GL_TRY(gl, DeleteBuffers(1, &staging));
    return {};
#endif  // !defined(__EMSCRIPTEN__)
}

}  // namespace dawn::native::opengl
