// Draws to the canvas in the way that the page's URL names, for
// surface-test.mjs, which checks what the page shows:
//
//   ?scene=draw          clears the canvas red in every frame
//   ?scene=skip          clears it red in the first frame, and in later frames
//                        gets the surface texture but submits nothing to it
//   ?scene=write         writes green to the surface texture with
//                        queue.writeTexture in every frame, without a submit
//
// With &present=no, it never calls Present, like programs written for
// Emdawnwebgpu. It prints "ready" after a few frames.
#include <emscripten/emscripten.h>

#include <cstdint>
#include <string>
#include <vector>

#include "test.h"

namespace {

using namespace test;

std::string scene;
bool present = true;
wgpu::Surface surface;
uint32_t width = 0;
uint32_t height = 0;
int frames = 0;

void ClearRed(wgpu::Texture texture) {
    wgpu::RenderPassColorAttachment attachment{
        .view = texture.CreateView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {1, 0, 0, 1},
    };
    wgpu::RenderPassDescriptor passDescriptor{.colorAttachmentCount = 1,
                                              .colorAttachments = &attachment};
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.BeginRenderPass(&passDescriptor).End();
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
}

void WriteGreen(wgpu::Texture texture) {
    std::vector<uint8_t> pixels;
    pixels.reserve(width * height * 4);
    for (uint32_t i = 0; i < width * height; i++) {
        pixels.insert(pixels.end(), {0, 255, 0, 255});
    }
    wgpu::TexelCopyTextureInfo destination{.texture = texture};
    wgpu::TexelCopyBufferLayout layout{.bytesPerRow = width * 4, .rowsPerImage = height};
    wgpu::Extent3D size{width, height};
    device.GetQueue().WriteTexture(&destination, pixels.data(), pixels.size(), &layout, &size);
}

bool Frame(double, void*) {
    wgpu::SurfaceTexture surfaceTexture;
    surface.GetCurrentTexture(&surfaceTexture);
    if (scene == "draw" || (scene == "skip" && frames == 0)) {
        ClearRed(surfaceTexture.texture);
    } else if (scene == "write") {
        WriteGreen(surfaceTexture.texture);
    }
    if (present) {
        surface.Present();
    }
    device.Tick();
    instance.ProcessEvents();
    if (++frames == 10) {
        std::printf("ready\n");
    }
    return !deviceLost;
}

std::string Parameter(const char* name) {
    std::string script = "new URLSearchParams(location.search).get('" + std::string(name) + "')";
    char* value = emscripten_run_script_string(script.c_str());
    return value != nullptr ? value : "";
}

}  // namespace

int main() {
    scene = Parameter("scene");
    present = Parameter("present") != "no";
    if (!Setup()) {
        return 1;
    }
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvas{{.selector = "#canvas"}};
    wgpu::SurfaceDescriptor surfaceDescriptor{.nextInChain = &canvas};
    surface = instance.CreateSurface(&surfaceDescriptor);
    wgpu::SurfaceCapabilities capabilities;
    surface.GetCapabilities(adapter, &capabilities);
    int canvasWidth = 0;
    int canvasHeight = 0;
    emscripten_get_canvas_element_size("#canvas", &canvasWidth, &canvasHeight);
    width = uint32_t(canvasWidth);
    height = uint32_t(canvasHeight);
    wgpu::SurfaceConfiguration configuration{
        .device = device,
        .format = capabilities.formats[0],
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopyDst,
        .width = width,
        .height = height,
    };
    surface.Configure(&configuration);
    emscripten_request_animation_frame_loop(Frame, nullptr);
    return 0;
}
