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
#include <array>
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
    Snorm8,
    Depth16,
    RG11B10Ufloat,
    RGB9E5Ufloat,
    // Stencil values, read from two RGBA8 textures holding one bit per channel.
    Stencil8,
};

struct Layout {
    Packing packing;
    uint32_t bytesPerTexel;
};

std::optional<Layout> GetLayout(wgpu::TextureFormat format, Aspect aspect) {
    if (aspect == Aspect::Depth) {
        switch (format) {
            case wgpu::TextureFormat::Depth16Unorm:
                return Layout{Packing::Depth16, 2};
            case wgpu::TextureFormat::Depth32Float:
            case wgpu::TextureFormat::Depth32FloatStencil8:
                return Layout{Packing::Float32, 4};
            default:
                return std::nullopt;
        }
    }
    if (aspect == Aspect::Stencil) {
        switch (format) {
            case wgpu::TextureFormat::Stencil8:
            case wgpu::TextureFormat::Depth24PlusStencil8:
            case wgpu::TextureFormat::Depth32FloatStencil8:
                return Layout{Packing::Stencil8, 1};
            default:
                return std::nullopt;
        }
    }
    if (aspect != Aspect::Color) {
        return std::nullopt;
    }
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
        // -128 reads as -1.0 like -127, and is copied as -127.
        case wgpu::TextureFormat::R8Snorm:
            return Layout{Packing::Snorm8, 1};
        case wgpu::TextureFormat::RG8Snorm:
            return Layout{Packing::Snorm8, 2};
        case wgpu::TextureFormat::RGBA8Snorm:
            return Layout{Packing::Snorm8, 4};
        case wgpu::TextureFormat::RG11B10Ufloat:
            return Layout{Packing::RG11B10Ufloat, 4};
        // Values with more than one encoding are copied in the one with the smallest exponent.
        case wgpu::TextureFormat::RGB9E5Ufloat:
            return Layout{Packing::RGB9E5Ufloat, 4};
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

// Encoders of the formats that GLSL ES 3.00 has no packing functions for. They only see values
// decoded from the format, so they need no rounding.
constexpr const char* kEncoders = R"(
// A float as an unsigned float with 5 exponent bits and |mantissaBits| mantissa bits.
uint ToUfloat(float f, uint mantissaBits) {
    if (!(f > 0.0)) {
        return isnan(f) ? (31u << mantissaBits) | 1u : 0u;
    }
    uint bits = floatBitsToUint(f);
    uint mantissa = bits & 0x7fffffu;
    int exponent = int(bits >> 23u) - 127 + 15;
    if (isinf(f) || exponent >= 31) {
        return 31u << mantissaBits;
    }
    if (exponent <= 0) {
        uint shift = 23u - mantissaBits + uint(1 - exponent);
        return shift > 31u ? 0u : (mantissa | 0x800000u) >> shift;
    }
    return (uint(exponent) << mantissaBits) | (mantissa >> (23u - mantissaBits));
}
// RGB in RGB9E5, as OpenGL ES 3.0's section 3.8.3.2 encodes it.
uint ToRGB9E5(vec3 rgb) {
    vec3 c = clamp(rgb, 0.0, 65408.0);
    float maxc = max(c.r, max(c.g, c.b));
    if (maxc == 0.0) {
        return 0u;
    }
    // floor(log2(maxc)), exactly.
    int log2Max = int(floatBitsToUint(maxc) >> 23u) - 127;
    int exponent = max(-16, log2Max) + 16;
    float scale = exp2(float(exponent - 24));
    if (uint(floor(maxc / scale + 0.5)) == 512u) {
        exponent += 1;
        scale *= 2.0;
    }
    uvec3 s = uvec3(floor(c / scale + 0.5));
    return s.r | (s.g << 9u) | (s.b << 18u) | (uint(exponent) << 27u);
}
)";

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
        case Packing::Snorm8:
            pack =
                "uvec4 c = uvec4(ivec4(round(clamp(v, -1.0, 1.0) * 127.0))) & 0xffu;\n"
                "return uvec4(c.x | (c.y << 8u) | (c.z << 16u) | (c.w << 24u), 0u, 0u, 0u);\n";
            break;
        case Packing::Depth16:
            pack = "return uvec4(uint(round(clamp(v.r, 0.0, 1.0) * 65535.0)), 0u, 0u, 0u);\n";
            break;
        case Packing::RG11B10Ufloat:
            pack =
                "return uvec4(ToUfloat(v.r, 6u) | (ToUfloat(v.g, 6u) << 11u) |\n"
                "             (ToUfloat(v.b, 5u) << 22u), 0u, 0u, 0u);\n";
            break;
        case Packing::RGB9E5Ufloat:
            pack = "return uvec4(ToRGB9E5(v.rgb), 0u, 0u, 0u);\n";
            break;
        case Packing::Stencil8:
            break;
    }

    std::string samplers;
    std::string texelWords;
    if (packing == Packing::Stencil8) {
        samplers =
            "uniform highp sampler2D lowBits;\n"
            "uniform highp sampler2D highBits;\n";
        texelWords =
            "ivec2 coord = origin.xy + ivec2(int(x), int(y));\n"
            "uvec4 l = uvec4(greaterThan(texelFetch(lowBits, coord, 0), vec4(0.5)));\n"
            "uvec4 h = uvec4(greaterThan(texelFetch(highBits, coord, 0), vec4(0.5)));\n"
            "return uvec4(l.x | (l.y << 1u) | (l.z << 2u) | (l.w << 3u) | (h.x << 4u) |\n"
            "             (h.y << 5u) | (h.z << 6u) | (h.w << 7u), 0u, 0u, 0u);\n";
    } else {
        samplers = "uniform highp " + prefix + SamplerType(target) + " src;\n";
        std::string fetch = target == GL_TEXTURE_2D
                                ? "texelFetch(src, origin.xy + ivec2(int(x), int(y)), 0)"
                                : "texelFetch(src, origin + ivec3(int(x), int(y), int(z)), 0)";
        texelWords = value + " v = " + fetch + ";\n" + pack;
    }

    return "#version 300 es\n"
           "precision highp float;\n"
           "precision highp int;\n" +
           samplers +
           "uniform ivec3 origin;\n"
           "uniform uvec3 copySize;\n"
           "flat out uint word;\n" +
           std::string(kEncoders) + "uvec4 TexelWords(uint x, uint y, uint z) {\n" + texelWords +
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
           "        uvec4 words = TexelWords(texel % copySize.x, (texel / copySize.x) % "
           "copySize.y,\n"
           "                                 texel / (copySize.x * copySize.y));\n"
           "        result |= ((words[inTexel / 4u] >> (8u * (inTexel % 4u))) & 0xffu) << (8u * "
           "b);\n"
           "    }\n"
           "    word = result;\n"
           "}\n";
}

// GLSL ES 3.00 programs need a fragment shader even when rasterization is discarded.
constexpr const char* kEmptyFragmentShader =
    "#version 300 es\n"
    "void main() {}\n";

// A triangle covering the viewport, drawn with no vertex attributes, that outputs 1.0.
constexpr const char* kFullscreenVertexShader =
    "#version 300 es\n"
    "void main() {\n"
    "    gl_Position = vec4(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0, 0.0, "
    "1.0);\n"
    "}\n";
constexpr const char* kOnesFragmentShader =
    "#version 300 es\n"
    "precision mediump float;\n"
    "out vec4 color;\n"
    "void main() {\n"
    "    color = vec4(1.0);\n"
    "}\n";

// Puts the stencil bits from the planes DrawStencilBitPlanes() draws back together.
constexpr const char* kStencilShadowFragmentShader =
    "#version 300 es\n"
    "precision highp float;\n"
    "precision highp int;\n"
    "uniform highp sampler2D lowBits;\n"
    "uniform highp sampler2D highBits;\n"
    "out uvec4 value;\n"
    "void main() {\n"
    "    ivec2 coord = ivec2(gl_FragCoord.xy);\n"
    "    uvec4 l = uvec4(greaterThan(texelFetch(lowBits, coord, 0), vec4(0.5)));\n"
    "    uvec4 h = uvec4(greaterThan(texelFetch(highBits, coord, 0), vec4(0.5)));\n"
    "    value = uvec4(l.x | (l.y << 1u) | (l.z << 2u) | (l.w << 3u) | (h.x << 4u) |\n"
    "                  (h.y << 5u) | (h.z << 6u) | (h.w << 7u), 0u, 0u, 0u);\n"
    "}\n";

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

// Links a program, which writes |varying| with transform feedback unless it is null.
ResultOrError<GLuint> LinkProgram(const OpenGLFunctions& gl,
                                  const std::string& vertexSource,
                                  const std::string& fragmentSource,
                                  const char* varying) {
    GLuint vertexShader;
    DAWN_TRY_ASSIGN(vertexShader, CompileShader(gl, GL_VERTEX_SHADER, vertexSource));
    GLuint fragmentShader;
    DAWN_TRY_ASSIGN(fragmentShader, CompileShader(gl, GL_FRAGMENT_SHADER, fragmentSource));

    GLuint program = DAWN_GL_TRY(gl, CreateProgram());
    DAWN_GL_TRY(gl, AttachShader(program, vertexShader));
    DAWN_GL_TRY(gl, AttachShader(program, fragmentShader));
    if (varying != nullptr) {
        DAWN_GL_TRY(gl, TransformFeedbackVaryings(program, 1, &varying, GL_INTERLEAVED_ATTRIBS));
    }
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
        return DAWN_UNRECOVERABLE_ERROR(
            "Linking the program for a texture-to-buffer copy failed:\n%s", log.data());
    }
    return program;
}

ResultOrError<GLuint> GetProgram(const OpenGLFunctions& gl,
                                 TextureToBufferTFCache& cache,
                                 const Layout& layout,
                                 GLenum target) {
    uint32_t key = (static_cast<uint32_t>(layout.packing) << 16) | (layout.bytesPerTexel << 8) |
                   (target == GL_TEXTURE_2D         ? 0
                    : target == GL_TEXTURE_2D_ARRAY ? 1
                                                    : 2);
    if (auto it = cache.programs.find(key); it != cache.programs.end()) {
        return it->second;
    }
    GLuint program;
    DAWN_TRY_ASSIGN(program,
                    LinkProgram(gl, VertexShader(layout.packing, layout.bytesPerTexel, target),
                                kEmptyFragmentShader, "word"));
    cache.programs.emplace(key, program);
    return program;
}

#if defined(__EMSCRIPTEN__)
MaybeError EnsureCommonObjects(const OpenGLFunctions& gl, TextureToBufferTFCache& cache) {
    if (cache.sampler == 0) {
        // Integer textures are incomplete with linear filtering, even for texelFetch().
        DAWN_GL_TRY(gl, GenSamplers(1, &cache.sampler));
        DAWN_GL_TRY(gl, SamplerParameteri(cache.sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        DAWN_GL_TRY(gl, SamplerParameteri(cache.sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        // A vertex array with no attributes, so that no vertex buffer limits the vertex count.
        DAWN_GL_TRY(gl, GenVertexArrays(1, &cache.vertexArray));
    }
    return {};
}

// Packs the |copySize| texels from |origin| of the texture(s) bound for |program|, which also
// sets the program's other uniforms, and copies them to |buffer| at |offset| with the given row
// and image strides.
MaybeError CopyWithTransformFeedback(const OpenGLFunctions& gl,
                                     TextureToBufferTFCache& cache,
                                     GLuint program,
                                     uint32_t bytesPerTexel,
                                     const Origin3D& origin,
                                     const Extent3D& copySize,
                                     Buffer* buffer,
                                     uint64_t offset,
                                     uint64_t bytesPerRow,
                                     uint64_t bytesPerImage) {
    // The copied bytes without row or image padding, rounded up to whole words.
    const uint64_t rowBytes = uint64_t(copySize.width) * bytesPerTexel;
    const uint64_t copyBytes = rowBytes * copySize.height * copySize.depthOrArrayLayers;
    const GLsizei wordCount = checked_cast<GLsizei>((copyBytes + 3) / 4);

    DAWN_GL_TRY(gl, UseProgram(program));
    GLint location = DAWN_GL_TRY(gl, GetUniformLocation(program, "origin"));
    DAWN_GL_TRY(gl, Uniform3i(location, dchecked_cast<GLint>(origin.x),
                              dchecked_cast<GLint>(origin.y), dchecked_cast<GLint>(origin.z)));
    location = DAWN_GL_TRY(gl, GetUniformLocation(program, "copySize"));
    DAWN_GL_TRY(gl,
                Uniform3ui(location, copySize.width, copySize.height, copySize.depthOrArrayLayers));
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

    // Copy each row to its place, leaving the padding between rows and images as it was.
    auto ForEachRow = [&](auto copyRow) -> MaybeError {
        for (uint32_t z = 0; z < copySize.depthOrArrayLayers; z++) {
            for (uint32_t y = 0; y < copySize.height; y++) {
                DAWN_TRY(copyRow((uint64_t(z) * copySize.height + y) * rowBytes,
                                 offset + z * bytesPerImage + y * bytesPerRow));
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
            buffer->NoteWrittenData(to, data.data() + from, rowBytes);
            DAWN_GL_TRY(gl, BufferSubData(GL_COPY_WRITE_BUFFER, checked_cast<GLintptr>(to),
                                          checked_cast<GLsizeiptr>(rowBytes), data.data() + from));
            return {};
        }));
        DAWN_GL_TRY(gl, BindBuffer(GL_COPY_WRITE_BUFFER, 0));
    } else {
        DAWN_GL_TRY(gl, BindBuffer(GL_COPY_READ_BUFFER, staging));
        DAWN_GL_TRY(gl, BindBuffer(GL_COPY_WRITE_BUFFER, buffer->GetHandle()));
        DAWN_TRY(ForEachRow([&](uint64_t from, uint64_t to) -> MaybeError {
            DAWN_GL_TRY(gl,
                        CopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                                          checked_cast<GLintptr>(from), checked_cast<GLintptr>(to),
                                          checked_cast<GLsizeiptr>(rowBytes)));
            return {};
        }));
        DAWN_GL_TRY(gl, BindBuffer(GL_COPY_READ_BUFFER, 0));
        DAWN_GL_TRY(gl, BindBuffer(GL_COPY_WRITE_BUFFER, 0));
    }
    DAWN_GL_TRY(gl, DeleteBuffers(1, &staging));
    return {};
}

// WebGL 2 shaders can't read stencil values. Draws bit k of the stencil values of |layer| of
// |mipLevel| of |texture| into channel k % 4 of planes[k / 4], two new RGBA8 textures the size
// of the level: a stencil test on bit k lets a draw write 1.0 where the bit is set.
MaybeError DrawStencilBitPlanes(const OpenGLFunctions& gl,
                                TextureToBufferTFCache& cache,
                                Texture* texture,
                                uint32_t mipLevel,
                                uint32_t layer,
                                GLuint planes[2]) {
    if (cache.stencilBitsProgram == 0) {
        DAWN_TRY_ASSIGN(cache.stencilBitsProgram,
                        LinkProgram(gl, kFullscreenVertexShader, kOnesFragmentShader, nullptr));
    }
    const Extent3D size =
        texture->GetMipLevelSingleSubresourcePhysicalSize(mipLevel, Aspect::Stencil);
    const GLsizei width = dchecked_cast<GLsizei>(size.width);
    const GLsizei height = dchecked_cast<GLsizei>(size.height);

    DAWN_GL_TRY(gl, GenTextures(2, planes));
    for (uint32_t i = 0; i < 2; i++) {
        DAWN_GL_TRY(gl, BindTexture(GL_TEXTURE_2D, planes[i]));
        DAWN_GL_TRY(gl, TexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height));
    }

    GLuint framebuffer = 0;
    DAWN_GL_TRY(gl, GenFramebuffers(1, &framebuffer));
    DAWN_GL_TRY(gl, BindFramebuffer(GL_FRAMEBUFFER, framebuffer));
    const GLenum attachment = texture->GetGLFormat().internalFormat == GL_STENCIL_INDEX8
                                  ? GL_STENCIL_ATTACHMENT
                                  : GL_DEPTH_STENCIL_ATTACHMENT;
    if (texture->GetGLTarget() == GL_TEXTURE_2D) {
        DAWN_GL_TRY(gl, FramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D,
                                             texture->GetTextureHandle(), mipLevel));
    } else if (texture->GetGLTarget() == GL_TEXTURE_CUBE_MAP) {
        DAWN_GL_TRY(gl, FramebufferTexture2D(GL_FRAMEBUFFER, attachment,
                                             GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer,
                                             texture->GetTextureHandle(), mipLevel));
    } else {
        DAWN_GL_TRY(gl, FramebufferTextureLayer(GL_FRAMEBUFFER, attachment,
                                                texture->GetTextureHandle(), mipLevel, layer));
    }

    DAWN_GL_TRY(gl, UseProgram(cache.stencilBitsProgram));
    DAWN_GL_TRY(gl, BindVertexArray(cache.vertexArray));
    DAWN_GL_TRY(gl, Viewport(0, 0, width, height));
    DAWN_GL_TRY(gl, Disable(GL_SCISSOR_TEST));
    DAWN_GL_TRY(gl, Disable(GL_BLEND));
    DAWN_GL_TRY(gl, Disable(GL_CULL_FACE));
    DAWN_GL_TRY(gl, Disable(GL_DEPTH_TEST));
    DAWN_GL_TRY(gl, Disable(GL_SAMPLE_ALPHA_TO_COVERAGE));
    DAWN_GL_TRY(gl, Enable(GL_STENCIL_TEST));
    DAWN_GL_TRY(gl, StencilOp(GL_KEEP, GL_KEEP, GL_KEEP));
    DAWN_GL_TRY(gl, StencilMask(0));
    for (uint32_t i = 0; i < 2; i++) {
        DAWN_GL_TRY(gl, FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                             planes[i], 0));
        DAWN_TRY(CheckFramebufferComplete(gl, GL_FRAMEBUFFER));
        static constexpr std::array<GLfloat, 4> kZero = {0, 0, 0, 0};
        DAWN_GL_TRY(gl, ColorMask(true, true, true, true));
        DAWN_GL_TRY(gl, ClearBufferfv(GL_COLOR, 0, kZero.data()));
        for (uint32_t channel = 0; channel < 4; channel++) {
            const GLuint bit = 1u << (i * 4 + channel);
            DAWN_GL_TRY(gl, ColorMask(channel == 0, channel == 1, channel == 2, channel == 3));
            DAWN_GL_TRY(gl, StencilFunc(GL_EQUAL, bit, bit));
            DAWN_GL_TRY(gl, DrawArrays(GL_TRIANGLES, 0, 3));
        }
    }

    // Draws set the rest again: pipelines their state and render passes the stencil function.
    DAWN_GL_TRY(gl, ColorMask(true, true, true, true));
    DAWN_GL_TRY(gl, Disable(GL_STENCIL_TEST));
    DAWN_GL_TRY(gl, Enable(GL_SCISSOR_TEST));
    DAWN_GL_TRY(gl, BindVertexArray(0));
    DAWN_GL_TRY(gl, BindFramebuffer(GL_FRAMEBUFFER, 0));
    DAWN_GL_TRY(gl, DeleteFramebuffers(1, &framebuffer));
    return {};
}
#endif  // defined(__EMSCRIPTEN__)

}  // namespace

bool UsesTransformFeedbackForTextureToBufferCopy(const Texture* texture, Aspect aspect) {
#if defined(__EMSCRIPTEN__)
    return GetLayout(texture->GetFormat().format, aspect).has_value() &&
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
    Layout layout = *GetLayout(texture->GetFormat().format, source.aspect);
    GLenum target = texture->GetGLTarget();
    TextureToBufferTFCache& cache = device->GetTextureToBufferTFCache();

    // Stencil programs read the bit planes, which are 2D textures.
    GLuint program;
    DAWN_TRY_ASSIGN(program,
                    GetProgram(gl, cache, layout,
                               layout.packing == Packing::Stencil8 ? GL_TEXTURE_2D : target));
    DAWN_TRY(EnsureCommonObjects(gl, cache));

    const TypedTexelBlockInfo& blockInfo = GetBlockInfo(source);
    const uint64_t bytesPerRow = blockInfo.ToBytes(destination.blocksPerRow);
    const uint64_t bytesPerImage =
        blockInfo.ToBytes(destination.blocksPerRow * destination.rowsPerImage);
    const Origin3D origin = {dchecked_cast<uint32_t>(source.origin.x),
                             dchecked_cast<uint32_t>(source.origin.y),
                             dchecked_cast<uint32_t>(source.origin.z)};

    if (layout.packing != Packing::Stencil8) {
        DAWN_GL_TRY(gl, UseProgram(program));
        DAWN_GL_TRY(gl, ActiveTexture(GL_TEXTURE0));
        DAWN_GL_TRY(gl, BindTexture(target, texture->GetTextureHandle()));
        // texelFetch() levels count from the base level. Binding a view sets both levels again.
        DAWN_GL_TRY(gl, TexParameteri(target, GL_TEXTURE_BASE_LEVEL, source.mipLevel));
        DAWN_GL_TRY(gl, TexParameteri(target, GL_TEXTURE_MAX_LEVEL, source.mipLevel));
        DAWN_GL_TRY(gl, BindSampler(0, cache.sampler));
        GLint location = DAWN_GL_TRY(gl, GetUniformLocation(program, "src"));
        DAWN_GL_TRY(gl, Uniform1i(location, 0));
        DAWN_TRY(CopyWithTransformFeedback(gl, cache, program, layout.bytesPerTexel, origin,
                                           copySize, buffer, destination.offset, bytesPerRow,
                                           bytesPerImage));
        DAWN_GL_TRY(gl, BindSampler(0, 0));
        return {};
    }

    // Stencil: one layer at a time, through bit planes.
    for (uint32_t layer = 0; layer < copySize.depthOrArrayLayers; layer++) {
        GLuint planes[2];
        DAWN_TRY(
            DrawStencilBitPlanes(gl, cache, texture, source.mipLevel, origin.z + layer, planes));
        DAWN_GL_TRY(gl, UseProgram(program));
        for (uint32_t i = 0; i < 2; i++) {
            DAWN_GL_TRY(gl, ActiveTexture(GL_TEXTURE0 + i));
            DAWN_GL_TRY(gl, BindTexture(GL_TEXTURE_2D, planes[i]));
            DAWN_GL_TRY(gl, BindSampler(i, cache.sampler));
        }
        GLint location = DAWN_GL_TRY(gl, GetUniformLocation(program, "lowBits"));
        DAWN_GL_TRY(gl, Uniform1i(location, 0));
        location = DAWN_GL_TRY(gl, GetUniformLocation(program, "highBits"));
        DAWN_GL_TRY(gl, Uniform1i(location, 1));
        DAWN_TRY(CopyWithTransformFeedback(
            gl, cache, program, 1, {origin.x, origin.y, 0}, {copySize.width, copySize.height, 1},
            buffer, destination.offset + layer * bytesPerImage, bytesPerRow, bytesPerImage));
        for (uint32_t i = 0; i < 2; i++) {
            DAWN_GL_TRY(gl, BindSampler(i, 0));
        }
        DAWN_GL_TRY(gl, ActiveTexture(GL_TEXTURE0));
        DAWN_GL_TRY(gl, DeleteTextures(2, planes));
    }
    return {};
#endif  // !defined(__EMSCRIPTEN__)
}

MaybeError UpdateStencilShadow(const OpenGLFunctions& gl,
                               Device* device,
                               Texture* texture,
                               const SubresourceRange& range) {
#if !defined(__EMSCRIPTEN__)
    DAWN_UNREACHABLE();
#else
    TextureToBufferTFCache& cache = device->GetTextureToBufferTFCache();
    DAWN_TRY(EnsureCommonObjects(gl, cache));
    if (cache.stencilShadowProgram == 0) {
        DAWN_TRY_ASSIGN(
            cache.stencilShadowProgram,
            LinkProgram(gl, kFullscreenVertexShader, kStencilShadowFragmentShader, nullptr));
    }

    const GLenum target = texture->GetGLTarget();
    GLuint shadow = texture->GetStencilShadow();
    if (shadow == 0) {
        const Extent3D size = texture->GetMipLevelSingleSubresourcePhysicalSize(0, Aspect::Stencil);
        DAWN_GL_TRY(gl, GenTextures(1, &shadow));
        DAWN_GL_TRY(gl, BindTexture(target, shadow));
        if (target == GL_TEXTURE_2D || target == GL_TEXTURE_CUBE_MAP) {
            DAWN_GL_TRY(gl, TexStorage2D(target, texture->GetNumMipLevels(), GL_R8UI, size.width,
                                         size.height));
        } else {
            DAWN_GL_TRY(gl, TexStorage3D(target, texture->GetNumMipLevels(), GL_R8UI, size.width,
                                         size.height, texture->GetArrayLayers()));
        }
        texture->SetStencilShadow(shadow);
    }

    for (uint32_t level = range.baseMipLevel; level < range.baseMipLevel + range.levelCount;
         level++) {
        const Extent3D size =
            texture->GetMipLevelSingleSubresourcePhysicalSize(level, Aspect::Stencil);
        for (uint32_t layer = range.baseArrayLayer; layer < range.baseArrayLayer + range.layerCount;
             layer++) {
            GLuint planes[2];
            DAWN_TRY(DrawStencilBitPlanes(gl, cache, texture, level, layer, planes));

            GLuint framebuffer = 0;
            DAWN_GL_TRY(gl, GenFramebuffers(1, &framebuffer));
            DAWN_GL_TRY(gl, BindFramebuffer(GL_FRAMEBUFFER, framebuffer));
            if (target == GL_TEXTURE_2D) {
                DAWN_GL_TRY(gl, FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                     GL_TEXTURE_2D, shadow, level));
            } else if (target == GL_TEXTURE_CUBE_MAP) {
                DAWN_GL_TRY(gl, FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                     GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer, shadow,
                                                     level));
            } else {
                DAWN_GL_TRY(gl, FramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                        shadow, level, layer));
            }
            DAWN_TRY(CheckFramebufferComplete(gl, GL_FRAMEBUFFER));

            DAWN_GL_TRY(gl, UseProgram(cache.stencilShadowProgram));
            DAWN_GL_TRY(gl, BindVertexArray(cache.vertexArray));
            for (uint32_t i = 0; i < 2; i++) {
                DAWN_GL_TRY(gl, ActiveTexture(GL_TEXTURE0 + i));
                DAWN_GL_TRY(gl, BindTexture(GL_TEXTURE_2D, planes[i]));
                DAWN_GL_TRY(gl, BindSampler(i, cache.sampler));
            }
            GLint location =
                DAWN_GL_TRY(gl, GetUniformLocation(cache.stencilShadowProgram, "lowBits"));
            DAWN_GL_TRY(gl, Uniform1i(location, 0));
            location = DAWN_GL_TRY(gl, GetUniformLocation(cache.stencilShadowProgram, "highBits"));
            DAWN_GL_TRY(gl, Uniform1i(location, 1));
            DAWN_GL_TRY(gl, Viewport(0, 0, dchecked_cast<GLsizei>(size.width),
                                     dchecked_cast<GLsizei>(size.height)));
            DAWN_GL_TRY(gl, Disable(GL_SCISSOR_TEST));
            DAWN_GL_TRY(gl, Disable(GL_BLEND));
            DAWN_GL_TRY(gl, Disable(GL_CULL_FACE));
            DAWN_GL_TRY(gl, Disable(GL_DEPTH_TEST));
            DAWN_GL_TRY(gl, Disable(GL_STENCIL_TEST));
            DAWN_GL_TRY(gl, Disable(GL_SAMPLE_ALPHA_TO_COVERAGE));
            DAWN_GL_TRY(gl, ColorMask(true, true, true, true));
            DAWN_GL_TRY(gl, DrawArrays(GL_TRIANGLES, 0, 3));

            // The render pass sets the rest again.
            DAWN_GL_TRY(gl, Enable(GL_SCISSOR_TEST));
            for (uint32_t i = 0; i < 2; i++) {
                DAWN_GL_TRY(gl, BindSampler(i, 0));
            }
            DAWN_GL_TRY(gl, ActiveTexture(GL_TEXTURE0));
            DAWN_GL_TRY(gl, BindVertexArray(0));
            DAWN_GL_TRY(gl, BindFramebuffer(GL_FRAMEBUFFER, 0));
            DAWN_GL_TRY(gl, DeleteFramebuffers(1, &framebuffer));
            DAWN_GL_TRY(gl, DeleteTextures(2, planes));
        }
    }
    return {};
#endif  // !defined(__EMSCRIPTEN__)
}

}  // namespace dawn::native::opengl
