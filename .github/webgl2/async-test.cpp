// Checks that callbacks that allow spontaneous calls are called without the
// program ticking the device or processing events, as browsers call them, and
// that callbacks that only allow ProcessEvents wait for it.
#include <emscripten/html5.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "test.h"

namespace {

using namespace test;

const std::vector<uint8_t> kPattern = {1, 2, 3, 4, 5, 6, 7, 8};

bool workDone = false;
bool mapped = false;
bool scopePopped = false;
bool processEventsMapped = false;
int framesAfterDone = 0;

wgpu::Buffer MakeBuffer(wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor descriptor{.usage = usage, .size = kPattern.size()};
    return device.CreateBuffer(&descriptor);
}

// Copies the pattern into a buffer that can be mapped for reading.
wgpu::Buffer MakeReadback() {
    wgpu::Buffer source = MakeBuffer(wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
    device.GetQueue().WriteBuffer(source, 0, kPattern.data(), kPattern.size());
    wgpu::Buffer readback = MakeBuffer(wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(source, 0, readback, 0, kPattern.size());
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
    return readback;
}

wgpu::Buffer spontaneousReadback;
wgpu::Buffer processEventsReadback;

void StartTests() {
    spontaneousReadback = MakeReadback();
    spontaneousReadback.MapAsync(
        wgpu::MapMode::Read, 0, kPattern.size(), wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            mapped = true;
            const char* name = "mapAsync completes without processing events";
            if (status != wgpu::MapAsyncStatus::Success) {
                Check(name, false, "mapAsync failed: " + ToString(message));
                return;
            }
            const void* data = spontaneousReadback.GetConstMappedRange(0, kPattern.size());
            Check(name, data != nullptr && std::memcmp(data, kPattern.data(), kPattern.size()) == 0,
                  "wrong contents");
            spontaneousReadback.Unmap();
        });

    device.GetQueue().OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowSpontaneous, [](wgpu::QueueWorkDoneStatus status, wgpu::StringView) {
            workDone = true;
            Check("onSubmittedWorkDone completes without processing events",
                  status == wgpu::QueueWorkDoneStatus::Success, "status is not Success");
        });

    device.PushErrorScope(wgpu::ErrorFilter::Validation);
    MakeBuffer(wgpu::BufferUsage::MapRead | wgpu::BufferUsage::MapWrite);
    device.PopErrorScope(wgpu::CallbackMode::AllowSpontaneous,
                         [](wgpu::PopErrorScopeStatus, wgpu::ErrorType type, wgpu::StringView) {
                             scopePopped = true;
                             Check("popErrorScope completes without processing events",
                                   type == wgpu::ErrorType::Validation, "no validation error");
                         });

    processEventsReadback = MakeReadback();
    processEventsReadback.MapAsync(wgpu::MapMode::Read, 0, kPattern.size(),
                                   wgpu::CallbackMode::AllowProcessEvents,
                                   [](wgpu::MapAsyncStatus, wgpu::StringView) {
                                       processEventsMapped = true;
                                   });

    // Only checks progress: never ticks the device or processes events until the end.
    emscripten_request_animation_frame_loop(
        [](double, void*) -> bool {
            if (!(workDone && mapped && scopePopped) && !deviceLost) {
                return true;
            }
            // Give the AllowProcessEvents callback a few frames in which it must not be called.
            if (++framesAfterDone < 5) {
                return true;
            }
            Check("an AllowProcessEvents callback waits for ProcessEvents", !processEventsMapped,
                  "called without ProcessEvents");
            instance.ProcessEvents();
            Check("ProcessEvents then calls it", processEventsMapped, "not called");
            Check("device stays alive without uncaptured errors",
                  !deviceLost && uncapturedErrors.empty(),
                  deviceLost ? "device lost" : uncapturedErrors);
            if (failures == 0) {
                std::printf("All tests passed\n");
            } else {
                std::printf("%d test(s) failed\n", failures);
            }
            return false;
        },
        nullptr);
}

}  // namespace

int main() {
    instance = wgpu::CreateInstance();
    wgpu::RequestAdapterOptions adapterOptions{
        .featureLevel = wgpu::FeatureLevel::Compatibility,
        .backendType = wgpu::BackendType::OpenGLES,
    };
    instance.RequestAdapter(
        &adapterOptions, wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::RequestAdapterStatus status, wgpu::Adapter result, wgpu::StringView message) {
            Check("requestAdapter completes without processing events",
                  status == wgpu::RequestAdapterStatus::Success, ToString(message));
            if (!result) {
                std::printf("%d test(s) failed\n", failures);
                return;
            }
            adapter = std::move(result);
            wgpu::DeviceDescriptor deviceDescriptor;
            deviceDescriptor.SetUncapturedErrorCallback(
                [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
                    uncapturedErrors += ToString(message) + "\n";
                });
            deviceDescriptor.SetDeviceLostCallback(
                wgpu::CallbackMode::AllowSpontaneous,
                [](const wgpu::Device&, wgpu::DeviceLostReason, wgpu::StringView message) {
                    deviceLost = true;
                    std::printf("device lost: %s\n", ToString(message).c_str());
                });
            adapter.RequestDevice(
                &deviceDescriptor, wgpu::CallbackMode::AllowSpontaneous,
                [](wgpu::RequestDeviceStatus status, wgpu::Device result,
                   wgpu::StringView message) {
                    Check("requestDevice completes without processing events",
                          status == wgpu::RequestDeviceStatus::Success, ToString(message));
                    if (!result) {
                        std::printf("%d test(s) failed\n", failures);
                        return;
                    }
                    device = std::move(result);
                    StartTests();
                });
        });
    return 0;
}
