// Shared by the test programs that run-tests.mjs runs. Each prints
// "PASS <name>" or "FAIL <name>: <why>" for each test, then "All tests passed"
// or "<n> test(s) failed".
#ifndef WEBGL2_TEST_H_
#define WEBGL2_TEST_H_

#include <emscripten/html5.h>
#include <webgpu/webgpu_cpp.h>

#include <cstdio>
#include <functional>
#include <string>
#include <utility>

namespace test {

inline wgpu::Instance instance;
inline wgpu::Adapter adapter;
inline wgpu::Device device;
inline bool deviceLost = false;
inline std::string uncapturedErrors;
inline int failures = 0;

inline std::string ToString(wgpu::StringView view) {
    return std::string(view.data, view.length == WGPU_STRLEN
                                      ? std::char_traits<char>::length(view.data)
                                      : view.length);
}

inline void Check(const char* name, bool passed, const std::string& why = "") {
    if (passed) {
        std::printf("PASS %s\n", name);
    } else {
        std::printf("FAIL %s: %s\n", name, why.c_str());
        failures++;
    }
}

// Checks that |error| is a validation error, and prints its first line.
inline void CheckRejected(const char* name, const std::string& error) {
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

// Gets a compatibility-mode OpenGL ES adapter and a device with the default limits that records
// uncaptured errors and device loss. Returns false after printing a failure if either fails.
inline bool Setup() {
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
        return false;
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
        return false;
    }
    return true;
}

// Ticks the device every animation frame until |done| returns true or the device is lost, then
// checks that the device is alive with no uncaptured errors and prints the summary. WebGL only
// signals fences between tasks, so GPU work completes in later frames.
inline void FinishWhen(std::function<bool()> done) {
    static std::function<bool()> isDone;
    isDone = std::move(done);
    emscripten_request_animation_frame_loop(
        [](double, void*) -> bool {
            device.Tick();
            instance.ProcessEvents();
            if (!isDone() && !deviceLost) {
                return true;
            }
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

}  // namespace test

#endif  // WEBGL2_TEST_H_
