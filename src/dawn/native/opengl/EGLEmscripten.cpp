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

// An EGL 1.4 implementation over Emscripten's WebGL 2 contexts, with just the
// parts that the OpenGL ES backend uses. It is handed to DisplayEGL as the
// eglGetProcAddress, so the backend runs unchanged on top of it.
//
// - One display. Contexts need no config (EGL_KHR_no_config_context) and no
//   surface (EGL_KHR_surfaceless_context).
// - A context is a WebGL 2 context on the canvas named by kCanvasSelector, and
//   a window surface is that canvas's default framebuffer. The browser shows
//   the framebuffer when control returns to it, so eglSwapBuffers does nothing.
// - EGL_KHR_fence_sync is WebGL 2 fences. WebGL only updates a fence between
//   tasks, even after glFinish(), so a fence created before a glFinish() is
//   reported as signaled, and a wait with a timeout calls glFinish() instead of
//   blocking. glFinish() is wrapped to count the calls.

#include "src/dawn/native/opengl/EGLEmscripten.h"

#include <emscripten/em_asm.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgl.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>

// GL entry points from Emscripten's WebGL library. They are declared here
// instead of including GLES3/gl3.h, which would clash with Dawn's GL headers.
extern "C" {
void* glFenceSync(unsigned int condition, unsigned int flags);
unsigned int glClientWaitSync(void* sync, unsigned int flags, unsigned long long timeout);
void glGetSynciv(void* sync, unsigned int pname, int bufSize, int* length, int* values);
void glDeleteSync(void* sync);
void glFinish();
}

namespace dawn::native::opengl {

namespace {

constexpr unsigned int kGLSyncGPUCommandsComplete = 0x9117;
constexpr unsigned int kGLSyncFlushCommandsBit = 0x0001;
constexpr unsigned int kGLAlreadySignaled = 0x911A;
constexpr unsigned int kGLTimeoutExpired = 0x911B;
constexpr unsigned int kGLConditionSatisfied = 0x911C;
constexpr unsigned int kGLSyncStatus = 0x9114;
constexpr unsigned int kGLSignaled = 0x9119;

// TODO: Let the embedder choose the canvas the context is created on.
constexpr const char* kCanvasSelector = "#canvas";

// Handles are small non-null integers cast to pointers.
EGLDisplay const kDisplay = reinterpret_cast<EGLDisplay>(1);
EGLConfig const kConfig = reinterpret_cast<EGLConfig>(1);

struct WindowSurface {
    std::string selector;
};

EGLint gError = EGL_SUCCESS;
EGLenum gAPI = EGL_OPENGL_ES_API;
EGLContext gCurrentContext = EGL_NO_CONTEXT;
EGLSurface gCurrentDraw = EGL_NO_SURFACE;
EGLSurface gCurrentRead = EGL_NO_SURFACE;

// How many times glFinish() has been called, and the count when each fence was created.
uint64_t gFinishCount = 0;
std::unordered_map<void*, uint64_t> gFinishCountAtFence;

void Finish() {
    glFinish();
    gFinishCount++;
}

// glFinish() for the backend, which completes every fence created before it.
void EGLAPIENTRY FinishGL() {
    Finish();
}

// Whether glFinish() has been called since |sync| was created.
bool FinishedSince(void* sync) {
    auto fence = gFinishCountAtFence.find(sync);
    return fence != gFinishCountAtFence.end() && fence->second < gFinishCount;
}

EGLBoolean Fail(EGLint error) {
    gError = error;
    return EGL_FALSE;
}

EGLBoolean Succeed() {
    gError = EGL_SUCCESS;
    return EGL_TRUE;
}

EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ToWebGL(EGLContext context) {
    return reinterpret_cast<EMSCRIPTEN_WEBGL_CONTEXT_HANDLE>(context);
}

// Whether two selectors name the same canvas element.
bool SameCanvas(const char* a, const char* b) {
    if (std::strcmp(a, b) == 0) {
        return true;
    }
    return MAIN_THREAD_EM_ASM_INT(
        {
            const a = document.querySelector(UTF8ToString($0));
            return a !== null && a === document.querySelector(UTF8ToString($1));
        },
        a, b);
}

// EGL 1.0

EGLBoolean EGLAPIENTRY ChooseConfig(EGLDisplay display,
                                    const EGLint* attribs,
                                    EGLConfig* configs,
                                    EGLint configSize,
                                    EGLint* numConfig) {
    if (display != kDisplay) {
        return Fail(EGL_BAD_DISPLAY);
    }
    // The canvas is RGBA8 with no depth or stencil.
    for (const EGLint* attrib = attribs; attrib != nullptr && attrib[0] != EGL_NONE;
         attrib += 2) {
        switch (attrib[0]) {
            case EGL_RED_SIZE:
            case EGL_GREEN_SIZE:
            case EGL_BLUE_SIZE:
            case EGL_ALPHA_SIZE:
                if (attrib[1] != EGL_DONT_CARE && attrib[1] > 8) {
                    *numConfig = 0;
                    return Succeed();
                }
                break;
            case EGL_DEPTH_SIZE:
            case EGL_STENCIL_SIZE:
                if (attrib[1] != EGL_DONT_CARE && attrib[1] > 0) {
                    *numConfig = 0;
                    return Succeed();
                }
                break;
            default:
                break;
        }
    }
    *numConfig = 1;
    if (configs != nullptr && configSize > 0) {
        configs[0] = kConfig;
    }
    return Succeed();
}

EGLBoolean EGLAPIENTRY CopyBuffers(EGLDisplay, EGLSurface, EGLNativePixmapType) {
    return Fail(EGL_BAD_NATIVE_PIXMAP);
}

EGLContext EGLAPIENTRY CreateContext(EGLDisplay display,
                                     EGLConfig,
                                     EGLContext shareContext,
                                     const EGLint*) {
    if (display != kDisplay) {
        Fail(EGL_BAD_DISPLAY);
        return EGL_NO_CONTEXT;
    }
    // WebGL contexts cannot share objects.
    if (shareContext != EGL_NO_CONTEXT) {
        Fail(EGL_BAD_MATCH);
        return EGL_NO_CONTEXT;
    }
    // The requested version and robustness are ignored: WebGL 2 is ES 3.0 and
    // always robust.
    EmscriptenWebGLContextAttributes attributes;
    emscripten_webgl_init_context_attributes(&attributes);
    attributes.majorVersion = 2;
    attributes.minorVersion = 0;
    attributes.alpha = true;
    attributes.depth = false;
    attributes.stencil = false;
    attributes.antialias = false;
    attributes.premultipliedAlpha = true;
    attributes.preserveDrawingBuffer = false;
    attributes.enableExtensionsByDefault = true;
    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context =
        emscripten_webgl_create_context(kCanvasSelector, &attributes);
    if (context <= 0) {
        Fail(EGL_BAD_ALLOC);
        return EGL_NO_CONTEXT;
    }
    Succeed();
    return reinterpret_cast<EGLContext>(context);
}

EGLSurface EGLAPIENTRY CreatePbufferSurface(EGLDisplay, EGLConfig, const EGLint*) {
    Fail(EGL_BAD_MATCH);
    return EGL_NO_SURFACE;
}

EGLSurface EGLAPIENTRY CreatePixmapSurface(EGLDisplay,
                                           EGLConfig,
                                           EGLNativePixmapType,
                                           const EGLint*) {
    Fail(EGL_BAD_NATIVE_PIXMAP);
    return EGL_NO_SURFACE;
}

EGLSurface EGLAPIENTRY CreateWindowSurface(EGLDisplay display,
                                           EGLConfig,
                                           EGLNativeWindowType window,
                                           const EGLint*) {
    if (display != kDisplay) {
        Fail(EGL_BAD_DISPLAY);
        return EGL_NO_SURFACE;
    }
    // The native window is the canvas's CSS selector.
    const char* selector = reinterpret_cast<const char*>(window);
    // Only the canvas the contexts render to can be presented to.
    if (selector == nullptr || !SameCanvas(selector, kCanvasSelector)) {
        Fail(EGL_BAD_NATIVE_WINDOW);
        return EGL_NO_SURFACE;
    }
    Succeed();
    return reinterpret_cast<EGLSurface>(new WindowSurface{selector});
}

EGLBoolean EGLAPIENTRY DestroyContext(EGLDisplay display, EGLContext context) {
    if (display != kDisplay) {
        return Fail(EGL_BAD_DISPLAY);
    }
    if (context == gCurrentContext) {
        gCurrentContext = EGL_NO_CONTEXT;
    }
    emscripten_webgl_destroy_context(ToWebGL(context));
    return Succeed();
}

EGLBoolean EGLAPIENTRY DestroySurface(EGLDisplay display, EGLSurface surface) {
    if (display != kDisplay) {
        return Fail(EGL_BAD_DISPLAY);
    }
    if (surface == EGL_NO_SURFACE) {
        return Fail(EGL_BAD_SURFACE);
    }
    delete reinterpret_cast<WindowSurface*>(surface);
    return Succeed();
}

EGLBoolean EGLAPIENTRY GetConfigAttrib(EGLDisplay display,
                                       EGLConfig config,
                                       EGLint attribute,
                                       EGLint* value) {
    if (display != kDisplay) {
        return Fail(EGL_BAD_DISPLAY);
    }
    if (config != kConfig) {
        return Fail(EGL_BAD_CONFIG);
    }
    switch (attribute) {
        case EGL_RED_SIZE:
        case EGL_GREEN_SIZE:
        case EGL_BLUE_SIZE:
        case EGL_ALPHA_SIZE:
            *value = 8;
            return Succeed();
        case EGL_DEPTH_SIZE:
        case EGL_STENCIL_SIZE:
            *value = 0;
            return Succeed();
        case EGL_SURFACE_TYPE:
            *value = EGL_WINDOW_BIT;
            return Succeed();
        case EGL_RENDERABLE_TYPE:
        case EGL_CONFORMANT:
            *value = EGL_OPENGL_ES2_BIT | EGL_OPENGL_ES3_BIT;
            return Succeed();
        default:
            return Fail(EGL_BAD_ATTRIBUTE);
    }
}

EGLBoolean EGLAPIENTRY GetConfigs(EGLDisplay display,
                                  EGLConfig* configs,
                                  EGLint configSize,
                                  EGLint* numConfig) {
    return ChooseConfig(display, nullptr, configs, configSize, numConfig);
}

EGLDisplay EGLAPIENTRY GetCurrentDisplay() {
    return gCurrentContext == EGL_NO_CONTEXT ? EGL_NO_DISPLAY : kDisplay;
}

EGLSurface EGLAPIENTRY GetCurrentSurface(EGLint readdraw) {
    return readdraw == EGL_READ ? gCurrentRead : gCurrentDraw;
}

EGLDisplay EGLAPIENTRY GetDisplay(EGLNativeDisplayType) {
    return kDisplay;
}

EGLint EGLAPIENTRY GetError() {
    EGLint error = gError;
    gError = EGL_SUCCESS;
    return error;
}

EGLBoolean EGLAPIENTRY Initialize(EGLDisplay display, EGLint* major, EGLint* minor) {
    if (display != kDisplay) {
        return Fail(EGL_BAD_DISPLAY);
    }
    if (major != nullptr) {
        *major = 1;
    }
    if (minor != nullptr) {
        *minor = 4;
    }
    return Succeed();
}

EGLBoolean EGLAPIENTRY MakeCurrent(EGLDisplay display,
                                   EGLSurface draw,
                                   EGLSurface read,
                                   EGLContext context) {
    if (display != kDisplay) {
        return Fail(EGL_BAD_DISPLAY);
    }
    if (context == EGL_NO_CONTEXT && (draw != EGL_NO_SURFACE || read != EGL_NO_SURFACE)) {
        return Fail(EGL_BAD_MATCH);
    }
    if (context != gCurrentContext) {
        if (emscripten_webgl_make_context_current(ToWebGL(context)) != EMSCRIPTEN_RESULT_SUCCESS) {
            return Fail(EGL_BAD_CONTEXT);
        }
    }
    gCurrentContext = context;
    gCurrentDraw = draw;
    gCurrentRead = read;
    return Succeed();
}

EGLBoolean EGLAPIENTRY QueryContext(EGLDisplay display,
                                    EGLContext,
                                    EGLint attribute,
                                    EGLint* value) {
    if (display != kDisplay) {
        return Fail(EGL_BAD_DISPLAY);
    }
    switch (attribute) {
        case EGL_CONTEXT_CLIENT_TYPE:
            *value = EGL_OPENGL_ES_API;
            return Succeed();
        case EGL_CONTEXT_CLIENT_VERSION:
            *value = 3;
            return Succeed();
        default:
            return Fail(EGL_BAD_ATTRIBUTE);
    }
}

const char* EGLAPIENTRY QueryString(EGLDisplay display, EGLint name) {
    if (display == EGL_NO_DISPLAY) {
        // No client extensions.
        Fail(EGL_BAD_DISPLAY);
        return nullptr;
    }
    if (display != kDisplay) {
        Fail(EGL_BAD_DISPLAY);
        return nullptr;
    }
    Succeed();
    switch (name) {
        case EGL_CLIENT_APIS:
            return "OpenGL_ES";
        case EGL_EXTENSIONS:
            return "EGL_KHR_create_context EGL_EXT_create_context_robustness EGL_KHR_fence_sync "
                   "EGL_KHR_surfaceless_context EGL_KHR_no_config_context";
        case EGL_VENDOR:
            return "Emscripten";
        case EGL_VERSION:
            return "1.4 WebGL 2";
        default:
            Fail(EGL_BAD_PARAMETER);
            return nullptr;
    }
}

EGLBoolean EGLAPIENTRY QuerySurface(EGLDisplay display,
                                    EGLSurface surface,
                                    EGLint attribute,
                                    EGLint* value) {
    if (display != kDisplay) {
        return Fail(EGL_BAD_DISPLAY);
    }
    if (surface == EGL_NO_SURFACE) {
        return Fail(EGL_BAD_SURFACE);
    }
    const WindowSurface* windowSurface = reinterpret_cast<const WindowSurface*>(surface);
    int width = 0;
    int height = 0;
    if (emscripten_get_canvas_element_size(windowSurface->selector.c_str(), &width, &height) !=
        EMSCRIPTEN_RESULT_SUCCESS) {
        return Fail(EGL_BAD_NATIVE_WINDOW);
    }
    switch (attribute) {
        case EGL_WIDTH:
            *value = width;
            return Succeed();
        case EGL_HEIGHT:
            *value = height;
            return Succeed();
        default:
            return Fail(EGL_BAD_ATTRIBUTE);
    }
}

EGLBoolean EGLAPIENTRY SwapBuffers(EGLDisplay display, EGLSurface surface) {
    if (display != kDisplay) {
        return Fail(EGL_BAD_DISPLAY);
    }
    if (surface == EGL_NO_SURFACE) {
        return Fail(EGL_BAD_SURFACE);
    }
    // The browser presents the canvas when control returns to it.
    return Succeed();
}

EGLBoolean EGLAPIENTRY Terminate(EGLDisplay display) {
    return display == kDisplay ? Succeed() : Fail(EGL_BAD_DISPLAY);
}

EGLBoolean EGLAPIENTRY WaitGL() {
    return Succeed();
}

EGLBoolean EGLAPIENTRY WaitNative(EGLint) {
    return Succeed();
}

// EGL 1.1

EGLBoolean EGLAPIENTRY BindTexImage(EGLDisplay, EGLSurface, EGLint) {
    return Fail(EGL_BAD_SURFACE);
}

EGLBoolean EGLAPIENTRY ReleaseTexImage(EGLDisplay, EGLSurface, EGLint) {
    return Fail(EGL_BAD_SURFACE);
}

EGLBoolean EGLAPIENTRY SurfaceAttrib(EGLDisplay, EGLSurface, EGLint, EGLint) {
    return Fail(EGL_BAD_ATTRIBUTE);
}

EGLBoolean EGLAPIENTRY SwapInterval(EGLDisplay display, EGLint) {
    // The browser presents at its own rate.
    return display == kDisplay ? Succeed() : Fail(EGL_BAD_DISPLAY);
}

// EGL 1.2

EGLBoolean EGLAPIENTRY BindAPI(EGLenum api) {
    if (api != EGL_OPENGL_ES_API) {
        return Fail(EGL_BAD_PARAMETER);
    }
    gAPI = api;
    return Succeed();
}

EGLenum EGLAPIENTRY QueryAPI() {
    return gAPI;
}

EGLSurface EGLAPIENTRY CreatePbufferFromClientBuffer(EGLDisplay,
                                                     EGLenum,
                                                     EGLClientBuffer,
                                                     EGLConfig,
                                                     const EGLint*) {
    Fail(EGL_BAD_PARAMETER);
    return EGL_NO_SURFACE;
}

EGLBoolean EGLAPIENTRY ReleaseThread() {
    return MakeCurrent(kDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

EGLBoolean EGLAPIENTRY WaitClient() {
    return Succeed();
}

// EGL 1.4

EGLContext EGLAPIENTRY GetCurrentContext() {
    return gCurrentContext;
}

// EGL_KHR_fence_sync

EGLSyncKHR EGLAPIENTRY CreateSyncKHR(EGLDisplay display, EGLenum type, const EGLint* attribs) {
    if (display != kDisplay) {
        Fail(EGL_BAD_DISPLAY);
        return EGL_NO_SYNC_KHR;
    }
    if (type != EGL_SYNC_FENCE_KHR || (attribs != nullptr && attribs[0] != EGL_NONE)) {
        Fail(EGL_BAD_ATTRIBUTE);
        return EGL_NO_SYNC_KHR;
    }
    if (gCurrentContext == EGL_NO_CONTEXT) {
        Fail(EGL_BAD_MATCH);
        return EGL_NO_SYNC_KHR;
    }
    void* sync = glFenceSync(kGLSyncGPUCommandsComplete, 0);
    if (sync == nullptr) {
        Fail(EGL_BAD_ALLOC);
        return EGL_NO_SYNC_KHR;
    }
    gFinishCountAtFence[sync] = gFinishCount;
    Succeed();
    return static_cast<EGLSyncKHR>(sync);
}

EGLBoolean EGLAPIENTRY DestroySyncKHR(EGLDisplay display, EGLSyncKHR sync) {
    if (display != kDisplay) {
        return Fail(EGL_BAD_DISPLAY);
    }
    gFinishCountAtFence.erase(sync);
    glDeleteSync(sync);
    return Succeed();
}

EGLint EGLAPIENTRY ClientWaitSyncKHR(EGLDisplay display,
                                     EGLSyncKHR sync,
                                     EGLint flags,
                                     EGLTimeKHR timeout) {
    if (display != kDisplay) {
        Fail(EGL_BAD_DISPLAY);
        return EGL_FALSE;
    }
    unsigned int glFlags = (flags & EGL_SYNC_FLUSH_COMMANDS_BIT_KHR) ? kGLSyncFlushCommandsBit : 0;
    // WebGL only accepts a timeout of 0.
    switch (glClientWaitSync(sync, glFlags, 0)) {
        case kGLAlreadySignaled:
        case kGLConditionSatisfied:
            Succeed();
            return EGL_CONDITION_SATISFIED_KHR;
        case kGLTimeoutExpired:
            if (FinishedSince(sync)) {
                Succeed();
                return EGL_CONDITION_SATISFIED_KHR;
            }
            if (timeout == 0) {
                Succeed();
                return EGL_TIMEOUT_EXPIRED_KHR;
            }
            // The fence cannot signal before this task ends. glFinish() completes
            // everything before it, which is what the caller waits for.
            Finish();
            Succeed();
            return EGL_CONDITION_SATISFIED_KHR;
        default:
            Fail(EGL_BAD_PARAMETER);
            return EGL_FALSE;
    }
}

EGLBoolean EGLAPIENTRY GetSyncAttribKHR(EGLDisplay display,
                                        EGLSyncKHR sync,
                                        EGLint attribute,
                                        EGLint* value) {
    if (display != kDisplay) {
        return Fail(EGL_BAD_DISPLAY);
    }
    switch (attribute) {
        case EGL_SYNC_TYPE_KHR:
            *value = EGL_SYNC_FENCE_KHR;
            return Succeed();
        case EGL_SYNC_CONDITION_KHR:
            *value = EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR;
            return Succeed();
        case EGL_SYNC_STATUS_KHR: {
            int status = 0;
            glGetSynciv(sync, kGLSyncStatus, 1, nullptr, &status);
            *value = static_cast<unsigned int>(status) == kGLSignaled || FinishedSince(sync)
                         ? EGL_SIGNALED_KHR
                         : EGL_UNSIGNALED_KHR;
            return Succeed();
        }
        default:
            return Fail(EGL_BAD_ATTRIBUTE);
    }
}

struct Proc {
    const char* name;
    __eglMustCastToProperFunctionPointerType proc;
};

#define PROC(name) {"egl" #name, reinterpret_cast<__eglMustCastToProperFunctionPointerType>(name)}

const Proc kProcs[] = {
    PROC(ChooseConfig),
    PROC(CopyBuffers),
    PROC(CreateContext),
    PROC(CreatePbufferSurface),
    PROC(CreatePixmapSurface),
    PROC(CreateWindowSurface),
    PROC(DestroyContext),
    PROC(DestroySurface),
    PROC(GetConfigAttrib),
    PROC(GetConfigs),
    PROC(GetCurrentDisplay),
    PROC(GetCurrentSurface),
    PROC(GetDisplay),
    PROC(GetError),
    PROC(Initialize),
    PROC(MakeCurrent),
    PROC(QueryContext),
    PROC(QueryString),
    PROC(QuerySurface),
    PROC(SwapBuffers),
    PROC(Terminate),
    PROC(WaitGL),
    PROC(WaitNative),
    PROC(BindTexImage),
    PROC(ReleaseTexImage),
    PROC(SurfaceAttrib),
    PROC(SwapInterval),
    PROC(BindAPI),
    PROC(QueryAPI),
    PROC(CreatePbufferFromClientBuffer),
    PROC(ReleaseThread),
    PROC(WaitClient),
    PROC(GetCurrentContext),
    PROC(CreateSyncKHR),
    PROC(DestroySyncKHR),
    PROC(ClientWaitSyncKHR),
    PROC(GetSyncAttribKHR),
};

#undef PROC

}  // anonymous namespace

__eglMustCastToProperFunctionPointerType EGLAPIENTRY EmscriptenEGLGetProcAddress(const char* name) {
    if (std::strncmp(name, "egl", 3) == 0) {
        for (const Proc& proc : kProcs) {
            if (std::strcmp(proc.name, name) == 0) {
                return proc.proc;
            }
        }
        return nullptr;
    }
    if (std::strcmp(name, "glFinish") == 0) {
        return reinterpret_cast<__eglMustCastToProperFunctionPointerType>(FinishGL);
    }
    // GL functions, which needs -sGL_ENABLE_GET_PROC_ADDRESS when linking.
    return reinterpret_cast<__eglMustCastToProperFunctionPointerType>(
        emscripten_webgl_get_proc_address(name));
}

}  // namespace dawn::native::opengl
