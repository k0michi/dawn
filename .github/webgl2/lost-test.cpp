// Checks what happens when the WebGL context is lost, with WEBGL_lose_context:
// the device is lost from the browser's event without the program calling
// Dawn, pending work completes instead of waiting forever, no adapter is found
// while the context is lost, and a new adapter and device work once the
// browser restores it.
#include <emscripten/em_asm.h>
#include <emscripten/html5.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>

#include "test.h"

namespace {

using namespace test;

constexpr uint32_t kColor = 0xFF336699;  // RGBA8 (0x99, 0x66, 0x33, 0xFF)

std::string lostMessage;
wgpu::Buffer readback;
bool mapDone = false;
wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Success;

// Runs |step| every animation frame until it returns true, then runs |next|. Gives up after
// |frames| frames with a failure named |name|.
void Wait(const char* name, int frames, std::function<bool()> step, std::function<void()> next) {
    struct Waiting {
        const char* name;
        int frames;
        std::function<bool()> step;
        std::function<void()> next;
    };
    emscripten_request_animation_frame_loop(
        [](double, void* data) -> bool {
            auto* waiting = static_cast<Waiting*>(data);
            if (!waiting->step() && --waiting->frames > 0) {
                return true;
            }
            if (waiting->frames <= 0) {
                Check(waiting->name, false, "timed out");
                std::printf("%d test(s) failed\n", failures);
            } else {
                waiting->next();
            }
            delete waiting;
            return false;
        },
        new Waiting{name, frames, std::move(step), std::move(next)});
}

void LoseContext() {
    EM_ASM({ Module.loseContextExtension.loseContext(); });
}

void RestoreContext() {
    EM_ASM({ Module.loseContextExtension.restoreContext(); });
}

bool IsContextLost() {
    return EM_ASM_INT({ return Module.canvas.getContext('webgl2').isContextLost(); });
}

// Requests an adapter and a device whose callbacks are called spontaneously. Returns whether
// there was an adapter.
bool RequestDevice() {
    adapter = nullptr;
    device = nullptr;
    deviceLost = false;
    lostMessage.clear();
    wgpu::RequestAdapterOptions adapterOptions{
        .featureLevel = wgpu::FeatureLevel::Compatibility,
        .backendType = wgpu::BackendType::OpenGLES,
    };
    instance.RequestAdapter(&adapterOptions, wgpu::CallbackMode::AllowProcessEvents,
                            [](wgpu::RequestAdapterStatus, wgpu::Adapter result, wgpu::StringView) {
                                adapter = std::move(result);
                            });
    instance.ProcessEvents();
    if (!adapter) {
        return false;
    }
    wgpu::DeviceDescriptor deviceDescriptor;
    deviceDescriptor.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
            uncapturedErrors += ToString(message) + "\n";
        });
    deviceDescriptor.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device&, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            if (reason == wgpu::DeviceLostReason::Destroyed ||
                reason == wgpu::DeviceLostReason::CallbackCancelled) {
                return;
            }
            deviceLost = true;
            lostMessage = ToString(message);
            std::printf("device lost: %s\n", lostMessage.c_str());
        });
    adapter.RequestDevice(&deviceDescriptor, wgpu::CallbackMode::AllowProcessEvents,
                          [](wgpu::RequestDeviceStatus, wgpu::Device result, wgpu::StringView) {
                              device = std::move(result);
                          });
    instance.ProcessEvents();
    return device != nullptr;
}

// Clears a texture to kColor, copies it into |readback| and maps it, with a spontaneous callback.
void ClearAndMap() {
    wgpu::TextureDescriptor textureDescriptor{
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
        .size = {1, 1},
        .format = wgpu::TextureFormat::RGBA8Unorm,
    };
    wgpu::Texture texture = device.CreateTexture(&textureDescriptor);
    wgpu::BufferDescriptor bufferDescriptor{
        .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, .size = 4};
    readback = device.CreateBuffer(&bufferDescriptor);

    wgpu::RenderPassColorAttachment attachment{
        .view = texture.CreateView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0x99 / 255.0, 0x66 / 255.0, 0x33 / 255.0, 1.0},
    };
    wgpu::RenderPassDescriptor pass{.colorAttachmentCount = 1, .colorAttachments = &attachment};
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.BeginRenderPass(&pass).End();
    wgpu::TexelCopyTextureInfo source{.texture = texture};
    wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = 256}, .buffer = readback};
    wgpu::Extent3D size = {1, 1};
    encoder.CopyTextureToBuffer(&source, &destination, &size);
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);

    mapDone = false;
    readback.MapAsync(wgpu::MapMode::Read, 0, 4, wgpu::CallbackMode::AllowSpontaneous,
                      [](wgpu::MapAsyncStatus status, wgpu::StringView) {
                          mapDone = true;
                          mapStatus = status;
                      });
}

bool ReadbackIsColor() {
    uint32_t pixel = 0;
    std::memcpy(&pixel, readback.GetConstMappedRange(0, 4), 4);
    readback.Unmap();
    return pixel == kColor;
}

void Finish() {
    Check("no uncaptured errors", uncapturedErrors.empty(), uncapturedErrors);
    if (failures == 0) {
        std::printf("All tests passed\n");
    } else {
        std::printf("%d test(s) failed\n", failures);
    }
}

// 4. Losing the context while a mapping is pending completes the mapping with an error.
void LoseWhileMapping() {
    ClearAndMap();
    LoseContext();
    Wait(
        "a pending mapAsync completes after the loss", 120, [] { return mapDone && deviceLost; },
        [] {
            Check("a pending mapAsync completes after the loss",
                  mapStatus != wgpu::MapAsyncStatus::Success, "the mapping succeeded");
            Check("the second device is lost", deviceLost);
            Finish();
        });
}

// 3. After the restore, a new adapter and device draw again.
void AfterRestore() {
    if (!RequestDevice()) {
        Check("a new device after the restore", false, "no adapter or device");
        Finish();
        return;
    }
    Check("a new device after the restore", true);
    ClearAndMap();
    Wait(
        "the new device draws", 120, [] { return mapDone; },
        [] {
            Check("the new device draws",
                  mapStatus == wgpu::MapAsyncStatus::Success && ReadbackIsColor(),
                  "wrong readback");
            LoseWhileMapping();
        });
}

// 2. While the context is lost there is no adapter. The lost device and its adapter are
// released, which must keep listening for the next loss on the canvas.
void WhileLost() {
    Check("the device is lost without the program calling Dawn", deviceLost);
    // An idle device makes no GL calls, so it learns of the loss from the event.
    Check("the loss is from the webglcontextlost event",
          lostMessage.find("WebGL context was lost") != std::string::npos, lostMessage);
    readback = nullptr;
    device = nullptr;
    adapter = nullptr;
    bool hasAdapter = RequestDevice();
    Check("no adapter while the context is lost", !hasAdapter, "found one");
    device = nullptr;
    adapter = nullptr;
    RestoreContext();
    Wait("the context is restored", 120, [] { return !IsContextLost(); }, AfterRestore);
}

// 1. A working device is lost while idle, without the program calling Dawn. The device stops
// ticking itself a few frames after its work completes.
void LoseWhileIdle() {
    static int frames = 0;
    Wait(
        "the device goes idle", 120, [] { return ++frames == 30; },
        [] {
            LoseContext();
            Wait("the device is lost without the program calling Dawn", 120,
                 [] { return deviceLost; }, WhileLost);
        });
}

}  // namespace

int main() {
    instance = wgpu::CreateInstance();
    if (!RequestDevice()) {
        std::printf("FAIL requestDevice\n1 test(s) failed\n");
        return 0;
    }
    // After the device has made the context with its attributes. The extension must be found
    // before the loss: a lost context has no extensions.
    EM_ASM({
        Module.loseContextExtension =
            Module.canvas.getContext('webgl2').getExtension('WEBGL_lose_context');
    });
    ClearAndMap();
    Wait(
        "the first device draws", 120, [] { return mapDone; },
        [] {
            Check("the first device draws",
                  mapStatus == wgpu::MapAsyncStatus::Success && ReadbackIsColor(),
                  "wrong readback");
            LoseWhileIdle();
        });
    return 0;
}
