// Checks that the WebGL 2 adapter reports what it can do, and that the device
// rejects what it cannot do with validation errors instead of failing in GL.
//
// Prints "PASS <name>" or "FAIL <name>: <why>" for each test, then
// "All tests passed" or "<n> test(s) failed".
#include <emscripten/html5.h>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

wgpu::Instance instance;
wgpu::Adapter adapter;
wgpu::Device device;
bool deviceLost = false;
std::string uncapturedErrors;
int failures = 0;

std::string ToString(wgpu::StringView view) {
    return std::string(view.data, view.length == WGPU_STRLEN ? std::char_traits<char>::length(view.data)
                                                             : view.length);
}

void Check(const char* name, bool passed, const std::string& why = "") {
    if (passed) {
        std::printf("PASS %s\n", name);
    } else {
        std::printf("FAIL %s: %s\n", name, why.c_str());
        failures++;
    }
}

// Checks that |error| is a validation error, and prints its first line.
void CheckRejected(const char* name, const std::string& error) {
    Check(name, !error.empty(), "no validation error");
    if (!error.empty()) {
        std::printf("  %s\n", error.substr(0, error.find('\n')).c_str());
    }
}

// Runs |body| in a validation error scope and returns the error message, or "" for none.
template <typename F>
std::string ValidationErrorOf(F body) {
    device.PushErrorScope(wgpu::ErrorFilter::Validation);
    body();
    std::string message;
    bool popped = false;
    device.PopErrorScope(wgpu::CallbackMode::AllowProcessEvents,
                         [&](wgpu::PopErrorScopeStatus, wgpu::ErrorType type, wgpu::StringView text) {
                             if (type != wgpu::ErrorType::NoError) {
                                 message = ToString(text);
                                 if (message.empty()) {
                                     message = "(validation error)";
                                 }
                             }
                             popped = true;
                         });
    instance.ProcessEvents();
    if (!popped) {
        return "(error scope did not pop)";
    }
    return message;
}

void TestLimits() {
    wgpu::Limits adapterLimits;
    adapter.GetLimits(&adapterLimits);
    wgpu::Limits deviceLimits;
    device.GetLimits(&deviceLimits);

    // WebGL 2 has no storage buffers, storage textures or compute, and caps strides at 255.
    struct Expected {
        const char* name;
        uint64_t adapterValue;
        uint64_t deviceValue;
        uint64_t max;
    };
    const Expected expected[] = {
        {"maxStorageBuffersPerShaderStage", adapterLimits.maxStorageBuffersPerShaderStage,
         deviceLimits.maxStorageBuffersPerShaderStage, 0},
        {"maxStorageTexturesPerShaderStage", adapterLimits.maxStorageTexturesPerShaderStage,
         deviceLimits.maxStorageTexturesPerShaderStage, 0},
        {"maxComputeInvocationsPerWorkgroup", adapterLimits.maxComputeInvocationsPerWorkgroup,
         deviceLimits.maxComputeInvocationsPerWorkgroup, 0},
        {"maxComputeWorkgroupsPerDimension", adapterLimits.maxComputeWorkgroupsPerDimension,
         deviceLimits.maxComputeWorkgroupsPerDimension, 0},
        {"maxVertexBufferArrayStride", adapterLimits.maxVertexBufferArrayStride,
         deviceLimits.maxVertexBufferArrayStride, 255},
    };
    for (const Expected& limit : expected) {
        std::string name = std::string("adapter ") + limit.name;
        Check(name.c_str(), limit.adapterValue <= limit.max,
              std::to_string(limit.adapterValue) + " > " + std::to_string(limit.max));
        name = std::string("device ") + limit.name;
        Check(name.c_str(), limit.deviceValue <= limit.adapterValue,
              std::to_string(limit.deviceValue) + " > adapter's " +
                  std::to_string(limit.adapterValue));
    }
}

void TestStorageBufferBinding() {
    std::string error = ValidationErrorOf([] {
        wgpu::BindGroupLayoutEntry entry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Fragment,
            .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
        };
        wgpu::BindGroupLayoutDescriptor descriptor{.entryCount = 1, .entries = &entry};
        device.CreateBindGroupLayout(&descriptor);
    });
    CheckRejected("storage buffer binding is rejected", error);
}

void TestComputePipeline() {
    std::string error = ValidationErrorOf([] {
        wgpu::ShaderSourceWGSL wgsl{{.code = "@compute @workgroup_size(1) fn main() {}"}};
        wgpu::ShaderModuleDescriptor shaderDescriptor{.nextInChain = &wgsl};
        wgpu::ComputePipelineDescriptor descriptor{
            .compute = {.module = device.CreateShaderModule(&shaderDescriptor),
                        .entryPoint = "main"},
        };
        device.CreateComputePipeline(&descriptor);
    });
    CheckRejected("compute pipeline is rejected", error);
}

void TestIndexBufferUsages() {
    std::string error = ValidationErrorOf([] {
        wgpu::BufferDescriptor descriptor{
            .usage = wgpu::BufferUsage::Index | wgpu::BufferUsage::Vertex, .size = 16};
        device.CreateBuffer(&descriptor);
    });
    CheckRejected("Index | Vertex buffer is rejected", error);

    error = ValidationErrorOf([] {
        wgpu::BufferDescriptor descriptor{
            .usage = wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst |
                     wgpu::BufferUsage::CopySrc,
            .size = 16};
        device.CreateBuffer(&descriptor);
    });
    Check("Index | CopyDst | CopySrc buffer is accepted", error.empty(), error);
}

// Copies between an index buffer and other buffers, which WebGL cannot do on the GPU.
bool copiesDone = false;

void StartIndexBufferCopies() {
    const uint16_t indices[] = {0, 1, 2, 2, 1, 3, 0, 0};
    wgpu::BufferDescriptor stagingDescriptor{
        .usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst, .size = sizeof(indices)};
    wgpu::Buffer staging = device.CreateBuffer(&stagingDescriptor);
    device.GetQueue().WriteBuffer(staging, 0, indices, sizeof(indices));
    wgpu::BufferDescriptor indexDescriptor{
        .usage = wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc,
        .size = sizeof(indices)};
    wgpu::Buffer index = device.CreateBuffer(&indexDescriptor);
    wgpu::Buffer back = device.CreateBuffer(&stagingDescriptor);

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(staging, 0, index, 0, sizeof(indices));
    encoder.CopyBufferToBuffer(index, 0, back, 0, sizeof(indices));
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
    device.GetQueue().OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowProcessEvents,
        [](wgpu::QueueWorkDoneStatus, wgpu::StringView) { copiesDone = true; });
}

// WebGL only signals fences between tasks, so the copies finish in a later frame.
bool Frame(double, void*) {
    device.Tick();
    instance.ProcessEvents();
    if (!copiesDone && !deviceLost) {
        return true;
    }
    Check("copies between index and other buffers", !deviceLost && uncapturedErrors.empty(),
          deviceLost ? "device lost" : uncapturedErrors);
    if (failures == 0) {
        std::printf("All tests passed\n");
    } else {
        std::printf("%d test(s) failed\n", failures);
    }
    return false;
}

}  // namespace

int main() {
    instance = wgpu::CreateInstance();
    wgpu::RequestAdapterOptions adapterOptions{
        .featureLevel = wgpu::FeatureLevel::Compatibility,
        .backendType = wgpu::BackendType::OpenGLES,
    };
    instance.RequestAdapter(
        &adapterOptions, wgpu::CallbackMode::AllowProcessEvents,
        [](wgpu::RequestAdapterStatus, wgpu::Adapter result, wgpu::StringView) {
            adapter = std::move(result);
        });
    instance.ProcessEvents();
    if (!adapter) {
        std::printf("FAIL requestAdapter\n1 test(s) failed\n");
        return 1;
    }

    wgpu::DeviceDescriptor deviceDescriptor;
    deviceDescriptor.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
            uncapturedErrors += ToString(message) + "\n";
        });
    deviceDescriptor.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowProcessEvents,
        [](const wgpu::Device&, wgpu::DeviceLostReason, wgpu::StringView message) {
            deviceLost = true;
            std::printf("device lost: %s\n", ToString(message).c_str());
        });
    adapter.RequestDevice(
        &deviceDescriptor, wgpu::CallbackMode::AllowProcessEvents,
        [](wgpu::RequestDeviceStatus, wgpu::Device result, wgpu::StringView) {
            device = std::move(result);
        });
    instance.ProcessEvents();
    if (!device) {
        std::printf("FAIL requestDevice\n1 test(s) failed\n");
        return 1;
    }

    TestLimits();
    TestStorageBufferBinding();
    TestComputePipeline();
    TestIndexBufferUsages();
    StartIndexBufferCopies();
    emscripten_request_animation_frame_loop(Frame, nullptr);
    return 0;
}
