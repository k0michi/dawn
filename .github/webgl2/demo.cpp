// The swift-game demo (a checkered, textured quad on a cornflower blue clear)
// in C++, drawn by dawn::native's OpenGL ES backend on a WebGL 2 canvas.
//
// With ?present=no in the page's URL, it never calls Present, like programs
// written for Emdawnwebgpu.
#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu_cpp.h>

#include <cstdio>
#include <vector>

namespace {

const char* kShader = R"(
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@group(0) @binding(0) var imageSampler: sampler;
@group(0) @binding(1) var imageTexture: texture_2d<f32>;

@vertex
fn vertexMain(
    @location(0) position: vec2f,
    @location(1) uv: vec2f
) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4f(position, 0.0, 1.0);
    output.uv = uv;
    return output;
}

@fragment
fn fragmentMain(@location(0) uv: vec2f) -> @location(0) vec4f {
    return textureSample(imageTexture, imageSampler, uv);
}
)";

void Print(const char* what, wgpu::StringView message) {
    std::printf("%s: %.*s\n", what, int(message.length), message.data);
}

struct Demo {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::Surface surface;
    wgpu::RenderPipeline pipeline;
    wgpu::BindGroup bindGroup;
    wgpu::Buffer vertexBuffer;
    wgpu::Buffer indexBuffer;
    int frames = 0;
    bool present = true;
};

Demo demo;

bool Frame(double, void*) {
    wgpu::SurfaceTexture surfaceTexture;
    demo.surface.GetCurrentTexture(&surfaceTexture);
    if (surfaceTexture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
        surfaceTexture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
        std::printf("getCurrentTexture failed: %d\n", int(surfaceTexture.status));
        return false;
    }
    wgpu::RenderPassColorAttachment attachment{
        .view = surfaceTexture.texture.CreateView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0x64 / 255.0, 0x95 / 255.0, 0xED / 255.0, 1.0},
    };
    wgpu::RenderPassDescriptor passDescriptor{.colorAttachmentCount = 1,
                                              .colorAttachments = &attachment};
    wgpu::CommandEncoder encoder = demo.device.CreateCommandEncoder();
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(demo.pipeline);
    pass.SetBindGroup(0, demo.bindGroup);
    pass.SetVertexBuffer(0, demo.vertexBuffer);
    pass.SetIndexBuffer(demo.indexBuffer, wgpu::IndexFormat::Uint16);
    pass.DrawIndexed(6);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    demo.queue.Submit(1, &commands);
    if (demo.present) {
        demo.surface.Present();
    }
    demo.instance.ProcessEvents();
    if (++demo.frames == 1) {
        std::printf("first frame drawn\n");
    }
    return true;
}

}  // namespace

int main() {
    demo.present = !emscripten_run_script_int(
        "new URLSearchParams(location.search).get('present') === 'no'");
    demo.instance = wgpu::CreateInstance();

    // The OpenGL ES backend only offers compatibility-mode adapters.
    wgpu::RequestAdapterOptions adapterOptions{
        .featureLevel = wgpu::FeatureLevel::Compatibility,
        .backendType = wgpu::BackendType::OpenGLES,
    };
    demo.instance.RequestAdapter(
        &adapterOptions, wgpu::CallbackMode::AllowProcessEvents,
        [](wgpu::RequestAdapterStatus status, wgpu::Adapter result, wgpu::StringView message) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                Print("requestAdapter", message);
            }
            demo.adapter = std::move(result);
        });
    demo.instance.ProcessEvents();
    if (!demo.adapter) {
        return 1;
    }
    wgpu::AdapterInfo info;
    demo.adapter.GetInfo(&info);
    Print("adapter", info.device);
    Print("description", info.description);

    wgpu::DeviceDescriptor deviceDescriptor;
    deviceDescriptor.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
            Print("error", message);
        });
    deviceDescriptor.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowProcessEvents,
        [](const wgpu::Device&, wgpu::DeviceLostReason, wgpu::StringView message) {
            Print("device lost", message);
        });
    demo.adapter.RequestDevice(
        &deviceDescriptor, wgpu::CallbackMode::AllowProcessEvents,
        [](wgpu::RequestDeviceStatus status, wgpu::Device result, wgpu::StringView message) {
            if (status != wgpu::RequestDeviceStatus::Success) {
                Print("requestDevice", message);
            }
            demo.device = std::move(result);
        });
    demo.instance.ProcessEvents();
    if (!demo.device) {
        return 1;
    }
    demo.queue = demo.device.GetQueue();

    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvas{{.selector = "#canvas"}};
    wgpu::SurfaceDescriptor surfaceDescriptor{.nextInChain = &canvas};
    demo.surface = demo.instance.CreateSurface(&surfaceDescriptor);
    wgpu::SurfaceCapabilities capabilities;
    demo.surface.GetCapabilities(demo.adapter, &capabilities);
    if (capabilities.formatCount == 0) {
        std::printf("surface has no formats\n");
        return 1;
    }
    wgpu::TextureFormat format = capabilities.formats[0];
    int width = 0;
    int height = 0;
    emscripten_get_canvas_element_size("#canvas", &width, &height);
    wgpu::SurfaceConfiguration configuration{
        .device = demo.device,
        .format = format,
        .width = uint32_t(width),
        .height = uint32_t(height),
        .alphaMode = wgpu::CompositeAlphaMode::Auto,
        .presentMode = wgpu::PresentMode::Fifo,
    };
    demo.surface.Configure(&configuration);
    std::printf("surface %dx%d format %d\n", width, height, int(format));

    // A 64x64 checkerboard of 8x8 squares, as in the Swift demo.
    std::vector<uint8_t> pixels;
    for (uint32_t y = 0; y < 64; y++) {
        for (uint32_t x = 0; x < 64; x++) {
            uint8_t value = ((x / 8) + (y / 8)) % 2 == 0 ? 0xFF : 0x00;
            pixels.insert(pixels.end(), {value, value, value, 0xFF});
        }
    }
    wgpu::TextureDescriptor imageDescriptor{
        .usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding,
        .size = {64, 64},
        .format = wgpu::TextureFormat::RGBA8Unorm,
    };
    wgpu::Texture image = demo.device.CreateTexture(&imageDescriptor);
    wgpu::TexelCopyTextureInfo destination{.texture = image};
    wgpu::TexelCopyBufferLayout layout{.bytesPerRow = 64 * 4, .rowsPerImage = 64};
    wgpu::Extent3D size{64, 64};
    demo.queue.WriteTexture(&destination, pixels.data(), pixels.size(), &layout, &size);

    wgpu::SamplerDescriptor samplerDescriptor{
        .magFilter = wgpu::FilterMode::Nearest,
        .minFilter = wgpu::FilterMode::Nearest,
    };
    wgpu::Sampler sampler = demo.device.CreateSampler(&samplerDescriptor);

    wgpu::ShaderSourceWGSL wgsl{{.code = kShader}};
    wgpu::ShaderModuleDescriptor shaderDescriptor{.nextInChain = &wgsl};
    wgpu::ShaderModule shader = demo.device.CreateShaderModule(&shaderDescriptor);
    wgpu::VertexAttribute attributes[] = {
        {.format = wgpu::VertexFormat::Float32x2, .offset = 0, .shaderLocation = 0},
        {.format = wgpu::VertexFormat::Float32x2, .offset = 8, .shaderLocation = 1},
    };
    wgpu::VertexBufferLayout vertexLayout{
        .arrayStride = 16, .attributeCount = 2, .attributes = attributes};
    wgpu::ColorTargetState colorTarget{.format = format};
    wgpu::FragmentState fragment{
        .module = shader, .entryPoint = "fragmentMain", .targetCount = 1, .targets = &colorTarget};
    wgpu::RenderPipelineDescriptor pipelineDescriptor{
        .vertex = {.module = shader,
                   .entryPoint = "vertexMain",
                   .bufferCount = 1,
                   .buffers = &vertexLayout},
        .fragment = &fragment,
    };
    demo.pipeline = demo.device.CreateRenderPipeline(&pipelineDescriptor);

    wgpu::BindGroupEntry entries[] = {
        {.binding = 0, .sampler = sampler},
        {.binding = 1, .textureView = image.CreateView()},
    };
    wgpu::BindGroupDescriptor bindGroupDescriptor{
        .layout = demo.pipeline.GetBindGroupLayout(0), .entryCount = 2, .entries = entries};
    demo.bindGroup = demo.device.CreateBindGroup(&bindGroupDescriptor);

    float vertices[] = {
        -0.5, 0.5,  0.0, 0.0,  //
        -0.5, -0.5, 0.0, 1.0,  //
        0.5,  0.5,  1.0, 0.0,  //
        0.5,  -0.5, 1.0, 1.0,  //
    };
    uint16_t indices[] = {0, 1, 2, 2, 1, 3};
    wgpu::BufferDescriptor vertexDescriptor{
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Vertex, .size = sizeof(vertices)};
    demo.vertexBuffer = demo.device.CreateBuffer(&vertexDescriptor);
    demo.queue.WriteBuffer(demo.vertexBuffer, 0, vertices, sizeof(vertices));
    wgpu::BufferDescriptor indexDescriptor{
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Index, .size = sizeof(indices)};
    demo.indexBuffer = demo.device.CreateBuffer(&indexDescriptor);
    demo.queue.WriteBuffer(demo.indexBuffer, 0, indices, sizeof(indices));

    emscripten_request_animation_frame_loop(Frame, nullptr);
    return 0;
}
