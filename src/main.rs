use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::device::{Device, Features, DeviceExtensions};
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage, CpuBufferPool};
use winit::{EventsLoop, WindowBuilder, Window, VirtualKeyCode, ElementState};
use vulkano_win::VkSurfaceBuild;
use vulkano::swapchain::{Swapchain, SurfaceTransform, PresentMode};
use std::sync::Arc;
use vulkano::framebuffer::{Framebuffer, Subpass, RenderPassAbstract, FramebufferAbstract};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::command_buffer::{DynamicState, AutoCommandBufferBuilder};
use vulkano::sync::GpuFuture;
use vulkano::image::SwapchainImage;
use vulkano::pipeline::viewport::Viewport;
use std::fs::File;
use cgmath::{Rad, Matrix3, Matrix4, Point3, Vector3, Deg};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::pipeline::input_assembly::IndexType;
use std::path::Path;
use vulkano::image::traits::ImageViewAccess;
use winit::dpi::LogicalPosition;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
}

struct Index {
    position: [u32; 3],
}

unsafe impl vulkano::pipeline::input_assembly::Index for Index {
    fn ty() -> IndexType {
        IndexType::U32
    }
}

#[derive(Clone)]
struct UniformBufferObject {
    //    model: Matrix4<f32>,
    view:  Matrix4<f32>,
    proj:  Matrix4<f32>,
}

fn main() {
    let extensions = vulkano_win::required_extensions();

    let instance = Instance::new(None, &extensions, None).expect("No vulkan available");

    // Take the first physical device
    let physical_device = PhysicalDevice::enumerate(&instance).next().expect("Couldn't find device");
    println!("{:?}", physical_device.name());

    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();
    let window = surface.window();

    window.grab_cursor(true);
    window.hide_cursor(true);

    // Find a queue family that supports graphical usage.
    let queue_family = physical_device.queue_families()
        .find(|&q| q.supports_graphics() && q.supports_compute() && q.supports_transfers() && surface.is_supported(q).unwrap_or(false))
        .expect("couldn't find a queue supporting graphics");

    println!("Queues: {:?}", queue_family.queues_count());

    let device_extensions = DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::none() };

    let (device, mut queues) = {
        Device::new(
            physical_device,
            &Features::none(),
            &device_extensions,
            [(queue_family, 0.5)].iter().cloned()
        ).expect("")
    };

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical_device).unwrap();

        let usage = caps.supported_usage_flags;

        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        let format = caps.supported_formats[0].0;

        let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else {
            return;
        };

        Swapchain::new(
            device.clone(), surface.clone(), caps.min_image_count, format, initial_dimensions, 1,
            usage, &queue, SurfaceTransform::Identity, alpha, PresentMode::Fifo, true, None
        ).unwrap()
    };

    vulkano::impl_vertex!(Vertex, position);

    let (mut models, mat) = tobj::load_obj(Path::new("./teapot.obj")).expect("couldn't load teapot");
    let mut model = models.first().expect("couldn't find teapot model");
    let mesh = model.clone().mesh;
    let indices = mesh.indices;
    let vertices = mesh.positions.chunks(3).map(|chunk| Vertex { position: [chunk[0], chunk[1], chunk[2]] });

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(), BufferUsage::all(), vertices.into_iter()
    ).unwrap();

    let index_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(), BufferUsage::all(), indices.into_iter()
    ).unwrap();

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shader.vert"
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shader.frag"
        }
    }

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    ).unwrap());

    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer::<Vertex>()
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_list()
//        .cull_mode_disabled()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
    );

    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None };

    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<GpuFuture>;

    let mut view = Matrix4::look_at(
        Point3::new(5.0,  5.0, 0.0),
        Point3::new(0.0,  0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
    );

    let mut perspective = cgmath::perspective(
        Rad::from(Deg(90.0)),
        images[0].dimensions().width() as f32 / images[0].dimensions().height() as f32,
        0.01,
        10.0,
    );

    perspective.y.y *= -1.0;

    let mut running = true;
    let mut previous_mouse = (0.0, 0.0);


    struct Movement {
        forward: ElementState,
        backward: ElementState,
        left: ElementState,
        right: ElementState,
    }

    let mut movement_state = Movement {
        forward: ElementState::Released,
        backward: ElementState::Released,
        left: ElementState::Released,
        right: ElementState::Released,
    };

    while running {

        previous_frame_end.cleanup_finished();

        //window.set_cursor_position(LogicalPosition { x: images[0].dimensions().width() as f64 / 2.0, y: images[0].dimensions().height() as f64 / 2.0 });

        let uniform_buffer_object = UniformBufferObject { view, proj: perspective };

        let uniform_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(), BufferUsage::all(), vec![uniform_buffer_object].into_iter()
        ).unwrap();

        let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_buffer(uniform_buffer.clone()).unwrap()
            .build().unwrap()
        );

        if movement_state.forward == ElementState::Pressed {
            view = Matrix4::from_translation(Vector3::new(0.0, 0.0, 0.1)) * view
        }
        if movement_state.backward == ElementState::Pressed {
            view = Matrix4::from_translation(Vector3::new(0.0, 0.0, -0.1)) * view
        }
        if movement_state.left == ElementState::Pressed {
            view = Matrix4::from_translation(Vector3::new(0.1, 0.0, 0.0)) * view
        }
        if movement_state.right == ElementState::Pressed {
            view = Matrix4::from_translation(Vector3::new(-0.1, 0.0, 0.0)) * view
        }

        events_loop.poll_events(|e| match e {
            winit::Event::WindowEvent { event, .. } => match event {
                winit::WindowEvent::KeyboardInput { input, .. } => {
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::W) => movement_state = Movement { forward: input.state, ..movement_state },
                        Some(VirtualKeyCode::S) => movement_state = Movement { backward: input.state, ..movement_state },
                        Some(VirtualKeyCode::A) => movement_state = Movement { left: input.state, ..movement_state },
                        Some(VirtualKeyCode::D) => movement_state = Movement { right: input.state, ..movement_state },
                        _ => {}
                    }
                }
                winit::WindowEvent::CloseRequested => {
                    running = false;
                }
                _ => {}
            },
            winit::Event::DeviceEvent { event, .. } => match event {
                winit::DeviceEvent::MouseMotion { delta, .. } => {
                    view = Matrix4::from_angle_y(Rad::from(Deg(delta.0 as f32 / 2.0))) * Matrix4::from_angle_x(Rad::from(Deg(delta.1 as f32 / 2.0))) * view;
//                    view =  * view;
                },
                _ => {}
            }
            _ => {},
        });

        let (image_num, acquire_future) = match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(e) => panic!("{:?}", e),
        };

        let clear_values = vec!([0.0, 0.0, 1.0, 1.0].into());

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap()
            .draw_indexed(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), index_buffer.clone(), (set.clone()), ())
            .unwrap()
            .end_render_pass()
            .unwrap()
            .build()
            .unwrap();

        let future = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(f) => previous_frame_end = Box::new(f) as Box<_>,
            Err(e) => {println!("{:?}", e); previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;},
        }

    }

}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions.width() as f32, dimensions.height() as f32],
        depth_range: 0.0 .. 1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .build().unwrap()
        ) as Arc<FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}
