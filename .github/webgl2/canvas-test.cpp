// Checks that the adapter draws to the canvas of its compatible surface, that
// it rejects surfaces on other canvases, and that without a compatible surface
// it draws to #canvas. Each adapter's device has its own WebGL context, so
// this also checks that two devices on two canvases don't disturb each other.
#include <emscripten/em_asm.h>

#include <cstdint>
#include <string>

#include "test.h"

namespace {

using namespace test;

struct Target {
    wgpu::Surface surface;
    wgpu::Adapter adapter;
    wgpu::Device device;
};

wgpu::Surface MakeSurface(const char* selector) {
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvas{{.selector = selector}};
    wgpu::SurfaceDescriptor descriptor{.nextInChain = &canvas};
    return instance.CreateSurface(&descriptor);
}

// Gets an adapter compatible with |compatibleSurface|, or with no surface if it is null, and a
// device that records uncaptured errors.
bool GetDevice(wgpu::Surface compatibleSurface, Target* target) {
    wgpu::RequestAdapterOptions options{
        .featureLevel = wgpu::FeatureLevel::Compatibility,
        .backendType = wgpu::BackendType::OpenGLES,
        .compatibleSurface = compatibleSurface,
    };
    instance.RequestAdapter(&options, wgpu::CallbackMode::AllowProcessEvents,
                            [target](wgpu::RequestAdapterStatus, wgpu::Adapter result,
                                     wgpu::StringView) { target->adapter = std::move(result); });
    instance.ProcessEvents();
    if (!target->adapter) {
        return false;
    }
    wgpu::DeviceDescriptor deviceDescriptor;
    deviceDescriptor.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
            uncapturedErrors += ToString(message) + "\n";
        });
    target->adapter.RequestDevice(
        &deviceDescriptor, wgpu::CallbackMode::AllowProcessEvents,
        [target](wgpu::RequestDeviceStatus, wgpu::Device result, wgpu::StringView) {
            target->device = std::move(result);
        });
    instance.ProcessEvents();
    return bool(target->device);
}

size_t FormatCount(wgpu::Surface surface, wgpu::Adapter adapter) {
    wgpu::SurfaceCapabilities capabilities;
    surface.GetCapabilities(adapter, &capabilities);
    return capabilities.formatCount;
}

wgpu::SurfaceConfiguration Configuration(wgpu::Device device) {
    return {
        .device = device,
        .format = wgpu::TextureFormat::RGBA8Unorm,
        .width = 64,
        .height = 64,
    };
}

// Clears the surface's texture to |color| and submits.
void Clear(const Target& target, wgpu::Color color) {
    wgpu::SurfaceTexture surfaceTexture;
    target.surface.GetCurrentTexture(&surfaceTexture);
    wgpu::RenderPassColorAttachment attachment{
        .view = surfaceTexture.texture.CreateView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = color,
    };
    wgpu::RenderPassDescriptor passDescriptor{.colorAttachmentCount = 1,
                                              .colorAttachments = &attachment};
    wgpu::CommandEncoder encoder = target.device.CreateCommandEncoder();
    encoder.BeginRenderPass(&passDescriptor).End();
    wgpu::CommandBuffer commands = encoder.Finish();
    target.device.GetQueue().Submit(1, &commands);
}

// The color in the middle of the canvas's drawing buffer, as 0xRRGGBBAA.
uint32_t CanvasColor(const char* selector) {
    return uint32_t(MAIN_THREAD_EM_ASM_INT(
        {
            const canvas = document.querySelector(UTF8ToString($0));
            const gl = canvas.getContext('webgl2');
            const pixel = new Uint8Array(4);
            const readFramebuffer = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING);
            gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
            gl.readPixels(canvas.width / 2, canvas.height / 2, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE,
                          pixel);
            gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFramebuffer);
            return ((pixel[0] << 24) | (pixel[1] << 16) | (pixel[2] << 8) | pixel[3]) >>> 0;
        },
        selector));
}

std::string Hex(uint32_t color) {
    char text[16];
    std::snprintf(text, sizeof(text), "%08x", color);
    return text;
}

}  // namespace

int main() {
    MAIN_THREAD_EM_ASM({
        const other = document.createElement('canvas');
        other.id = 'other';
        other.width = 64;
        other.height = 64;
        document.body.appendChild(other);
    });
    instance = wgpu::CreateInstance();

    Target other{.surface = MakeSurface("#other")};
    if (!GetDevice(other.surface, &other)) {
        std::printf("FAIL adapter for #other\n1 test(s) failed\n");
        return 1;
    }
    Check("the compatible surface's canvas has formats", FormatCount(other.surface, other.adapter) > 0,
          "no formats");

    wgpu::Surface canvasSurface = MakeSurface("#canvas");
    Check("another canvas has no formats", FormatCount(canvasSurface, other.adapter) == 0,
          "has formats");
    device = other.device;
    wgpu::SurfaceConfiguration wrongConfiguration = Configuration(other.device);
    CheckRejected("configuring another canvas is rejected",
                  ValidationErrorOf([&] { canvasSurface.Configure(&wrongConfiguration); }));

    Target fallback{.surface = canvasSurface};
    if (!GetDevice(nullptr, &fallback)) {
        std::printf("FAIL adapter without a compatible surface\n%d test(s) failed\n", failures + 1);
        return 1;
    }
    Check("without a compatible surface, #canvas has formats",
          FormatCount(fallback.surface, fallback.adapter) > 0, "no formats");

    wgpu::SurfaceConfiguration otherConfiguration = Configuration(other.device);
    other.surface.Configure(&otherConfiguration);
    wgpu::SurfaceConfiguration fallbackConfiguration = Configuration(fallback.device);
    fallback.surface.Configure(&fallbackConfiguration);

    // Draw to both, then read both: each device must have drawn to its own canvas.
    Clear(other, {1, 0, 0, 1});
    Clear(fallback, {0, 0, 1, 1});
    uint32_t otherColor = CanvasColor("#other");
    uint32_t canvasColor = CanvasColor("#canvas");
    Check("#other shows what its device drew", otherColor == 0xff0000ff, "got " + Hex(otherColor));
    Check("#canvas shows what its device drew", canvasColor == 0x0000ffff, "got " + Hex(canvasColor));

    Check("no uncaptured errors", uncapturedErrors.empty(), uncapturedErrors);
    if (failures == 0) {
        std::printf("All tests passed\n");
    } else {
        std::printf("%d test(s) failed\n", failures);
    }
    return 0;
}
