#![feature(crate_visibility_modifier)]
#![feature(vec_remove_item)]
#![feature(duration_float)]

use vulkano::device::{Device, Queue};
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage, ImmutableBuffer, CpuBufferPool};
use winit::{Window, VirtualKeyCode, ElementState, EventsLoop, ControlFlow, Event};
use std::sync::{Arc, RwLock};
use vulkano::framebuffer::{Framebuffer, Subpass, RenderPassAbstract, FramebufferAbstract};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::command_buffer::{DynamicState, AutoCommandBufferBuilder};
use vulkano::sync::GpuFuture;
use vulkano::image::{SwapchainImage, AttachmentImage};
use vulkano::pipeline::viewport::Viewport;
use cgmath::{Rad, Matrix3, Matrix4, Point3, Vector3, Deg, Euler, Quaternion, Decomposed, Basis3, vec3, Vector4};
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, DescriptorSetsCollection, FixedSizeDescriptorSetsPool};
use std::path::Path;
use vulkano::image::traits::ImageViewAccess;
use crate::camera::Camera;
use vulkano::format::Format;
use frozengame::{FrozenGameBuilder};
use fuji::{FujiBuilder};
use frozengame::model::{Vertex, RenderDrawable, Mesh};
use specs::prelude::*;
use crate::components::{Position, RenderSystem, MeshBuffer, MeshUniformSystem, MovementSystem, PressedKeys, EventSystem};
use crate::components::player::{ActivePlayer};
use vulkano::instance::QueueFamily;
use tobj::load_obj;
use vulkano::swapchain::{Swapchain, Surface};
use std::cell::RefCell;
use vulkano::descriptor::PipelineLayoutAbstract;
use std::time::{Duration, Instant};

mod camera;
mod components;

#[derive(Clone)]
pub struct UniformBufferObject {
    model: Matrix4<f32>,
    view:  Matrix4<f32>,
    proj:  Matrix4<f32>,
}

extern crate frozengame;

fn main() {
    let mut fuji = FujiBuilder::new()
        .with_window()
        .build().unwrap()
        .with_graphics_queue()
        .with_present_queue()
        .with_swapchain()
        .build().unwrap();

    fuji.create_swapchain();

    let mut world = World::new();
    world.register::<Position>();
    world.register::<Camera<f32>>();
    world.register::<components::GraphicsPipeline>();
    world.register::<components::MeshBuffer<Vertex, u32>>();
    world.register::<components::MeshDescriptorSet>();
    world.register::<components::DescriptorSetPool>();

    let self_player = world.create_entity().with(Position::default()).with(Camera::default()).build();
    world.add_resource(ActivePlayer(self_player));
    world.add_resource(Duration::from_secs(0));

    let engine_instance = FrozenGameBuilder::new(fuji).build();

    let fuji = Box::leak(Box::new(engine_instance.fuji.clone()));

    let swapchain = fuji.swapchain();
    let images = fuji.swapchain_images();
    let events_loop: &RwLock<EventsLoop> = fuji.events_loop();
    let queue: &Arc<Queue> = fuji.graphics_queue();
    let device = fuji.device();

    world.add_resource(CpuBufferPool::new(device.clone(), BufferUsage::all()) as CpuBufferPool<UniformBufferObject>);

    mod teapot_vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/teapot_shader.vert"
        }
    }

    mod teapot_fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/teapot_shader.frag"
        }
    }

    let teapot_vs = teapot_vs::Shader::load(device.clone()).unwrap();
    let teapot_fs = teapot_fs::Shader::load(device.clone()).unwrap();

    mod cube_vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/cube_shader.vert"
        }
    }

    mod cube_fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/cube_shader.frag"
        }
    }

    let cube_vs = cube_vs::Shader::load(device.clone()).unwrap();
    let cube_fs = cube_fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16Unorm,
                samples: 1,
            }
        },
        pass: {
            color: [ color ],
            depth_stencil: { depth }
        }
    ).unwrap());

    let teapot_pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer::<Vertex>()
        .vertex_shader(teapot_vs.main_entry_point(), ())
        .triangle_list()
        .depth_stencil_simple_depth()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(teapot_fs.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
    );

    let (models, materials) = load_obj(&Path::new("./teapot.obj")).unwrap();

    let mesh = Mesh::from(models.first().unwrap().mesh.clone());

    let mut immutable_indices_buf = CpuAccessibleBuffer::from_iter(
        device.clone(), BufferUsage::all(), mesh.indices.into_iter()
    ).unwrap();

    let mut immutable_vert_buf = CpuAccessibleBuffer::from_iter(
        device.clone(), BufferUsage::all(), mesh.vertices.into_iter(),
    ).unwrap();

    use frozengame::model::Vertex;

    // teapot object
    world.create_entity()
        .with(MeshBuffer::<Vertex, u32>::from(vec![immutable_vert_buf.clone()], immutable_indices_buf.clone()))
        .with(components::GraphicsPipeline(teapot_pipeline.clone()))
        .with(components::DescriptorSetPool(FixedSizeDescriptorSetsPool::new(teapot_pipeline.clone(), 0)))
        .with(Position::default())
        .build();

    let mut dynamic_state = DynamicState::default();

    let mut framebuffers = window_size_dependent_setup(device.clone(), &images, render_pass.clone(), &mut dynamic_state);

    world.add_resource::<Option<Arc<Swapchain<winit::Window>>>>(Some(swapchain.clone()));
    world.add_resource::<Vec<Arc<FramebufferAbstract + Send + Sync>>>(framebuffers);
    world.add_resource::<Arc<Device>>(device.clone());
    world.add_resource::<Option<Arc<Queue>>>(Some(queue.clone()));

    let mut dispatcher = DispatcherBuilder::new()
        .with(MeshUniformSystem, "mesh system", &[])
        .with(RenderSystem { previous_frame: None }, "render system", &[])
        .with(MovementSystem, "movement system", &[])
        .with(EventSystem, "event system", &[])
//        .with(Camera::<f32>::default(), "camera system", &[])
        .build();

//    world.add_resource(events_loop.to_owned());
    world.add_resource(dynamic_state);
    dispatcher.setup(&mut world.res);
    dispatcher.dispatch(&mut world.res);
    world.maintain();

    #[derive(Clone, Copy)]
    struct Movement {
        forward:  ElementState,
        backward: ElementState,
        left:     ElementState,
        right:    ElementState,
        up:       ElementState,
        down:     ElementState,
    }

    let mut movement_state = Movement {
        forward:  ElementState::Released,
        backward: ElementState::Released,
        left:     ElementState::Released,
        right:    ElementState::Released,
        up:       ElementState::Released,
        down:     ElementState::Released,
    };

    loop {
        world.add_resource::<Option<Event>>(None);
        if let Ok(ref mut events_loop) = events_loop.write() {
            events_loop.poll_events(|event| {
                world.add_resource(Some(event));
//                match event {
//                    winit::Event::WindowEvent { event, .. } => match event {
//                        winit::WindowEvent::KeyboardInput { input, .. } => {
//                            println!("pressed movement key");
//                            match input.state {
//                                ElementState::Pressed => {
//                                    input.virtual_keycode.map(|r| if !pressed_keys.contains(&r) {
//                                        pressed_keys.push(r)
//                                    });
//                                },
//                                ElementState::Released => {
//                                    input.virtual_keycode.map(|r| pressed_keys.remove_item(&r).expect("removed key that didn't exist"));
//                                },
//                            }
//                        }
//                        winit::WindowEvent::CloseRequested => {
//                            run = false;
//                        }
//                        _ => {}
//                    },
//                    winit::Event::DeviceEvent { event, .. } => match event {
//                        winit::DeviceEvent::MouseMotion { delta, .. } => {
////                            camera.turn(Deg(delta.0 as f32 * 0.1));
////                            camera.pitch(Deg(delta.1 as f32 * 0.1));
//                        },
//                        _ => {}
//                    }
//                    _ => {},
//                };
            });
        }

        dispatcher.dispatch(&world.res);
    }

//    events_loop.write().unwrap().run_forever(|event| {
//        match event {
//            winit::Event::WindowEvent { event, .. } => match event {
//                winit::WindowEvent::KeyboardInput { input, .. } => {
//                    println!("pressed movement key");
//                    input.virtual_keycode.map(|r| pressed_keys.push(r));
//                }
//                winit::WindowEvent::CloseRequested => {
//                    return ControlFlow::Break;
//                }
//                _ => {}
//            },
//            winit::Event::DeviceEvent { event, .. } => match event {
//                winit::DeviceEvent::MouseMotion { delta, .. } => {
////                    camera.turn(Deg(delta.0 as f32 * 0.1));
////                    camera.pitch(Deg(delta.1 as f32 * 0.1));
//                },
//                _ => {}
//            }
//            _ => {},
//        };
//        world.add_resource(PressedKeys(pressed_keys));
//        world.add_resource(movement_state);
//
//        ControlFlow::Continue
//    });

//    let teapot = engine_instance.build_model(teapot_pipeline.clone())
//        .with_obj_path(&mut Path::new("./teapot.obj"))
//        .build().unwrap();
//
//    let cube_pipeline = Arc::new(GraphicsPipeline::start()
//        .vertex_input_single_buffer::<Vertex>()
//        .vertex_shader(cube_vs.main_entry_point(), ())
//        .triangle_list()
//        .depth_stencil_simple_depth()
//        .viewports_dynamic_scissors_irrelevant(1)
//        .fragment_shader(cube_fs.main_entry_point(), ())
//        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
//        .build(device.clone())
//        .unwrap()
//    );
//
//    let cube = engine_instance.build_model(cube_pipeline.clone())
//        .with_obj_path(&mut Path::new("./cube.obj"))
//        .build().unwrap();
//
//    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None };
//
//    let framebuffers = window_size_dependent_setup(device.clone(), &images, render_pass.clone(), &mut dynamic_state);
//
//    let _recreate_swapchain = false;
//
//    let mut previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<GpuFuture>;
//
//    let mut perspective = cgmath::perspective(
//        Rad::from(Deg(45.0)),
//        images[0].dimensions().width() as f32 / images[0].dimensions().height() as f32,
//        0.01,
//        100.0,
//    );
//
//    perspective.y.y *= -1.0;
//
//    let mut running = true;
//    let _previous_mouse = (0.0, 0.0);
//
//    struct Movement {
//        forward:  ElementState,
//        backward: ElementState,
//        left:     ElementState,
//        right:    ElementState,
//        up:       ElementState,
//        down:     ElementState,
//    }
//
//    let mut movement_state = Movement {
//        forward:  ElementState::Released,
//        backward: ElementState::Released,
//        left:     ElementState::Released,
//        right:    ElementState::Released,
//        up:       ElementState::Released,
//        down:     ElementState::Released,
//    };
//
//    let mut camera: Camera<f32> = Camera::default();
//
//    while running {
//        previous_frame_end.cleanup_finished();
//
//        if movement_state.forward == ElementState::Pressed {
//            camera.move_forward(0.1);
//        }
//        if movement_state.backward == ElementState::Pressed {
//            camera.move_forward(-0.1);
//        }
//        if movement_state.left == ElementState::Pressed {
//            camera.move_left(0.1);
//        }
//        if movement_state.right == ElementState::Pressed {
//            camera.move_left(-0.1);
//        }
//        if movement_state.up == ElementState::Pressed {
//            camera.move_up(0.1);
//        }
//        if movement_state.down == ElementState::Pressed {
//            camera.move_up(-0.1);
//        }
//
//        if let Ok(ref mut e) = events_loop.write() {
//            e.poll_events(|e| match e {
//                winit::Event::WindowEvent { event, .. } => match event {
//                    winit::WindowEvent::KeyboardInput { input, .. } => {
//                        match input.virtual_keycode {
//                            Some(VirtualKeyCode::W) => movement_state = Movement { forward:  input.state, ..movement_state },
//                            Some(VirtualKeyCode::S) => movement_state = Movement { backward: input.state, ..movement_state },
//                            Some(VirtualKeyCode::A) => movement_state = Movement { left:     input.state, ..movement_state },
//                            Some(VirtualKeyCode::D) => movement_state = Movement { right:    input.state, ..movement_state },
//                            Some(VirtualKeyCode::Space) => movement_state = Movement { up:    input.state, ..movement_state },
//                            Some(VirtualKeyCode::LControl) => movement_state = Movement { down:    input.state, ..movement_state },
//                            Some(VirtualKeyCode::Escape) => running = false,
//                            _ => {}
//                        }
//                    }
//                    winit::WindowEvent::CloseRequested => {
//                        running = false;
//                    }
//                    _ => {}
//                },
//                winit::Event::DeviceEvent { event, .. } => match event {
//                    winit::DeviceEvent::MouseMotion { delta, .. } => {
//                        camera.turn(Deg(delta.0 as f32 * 0.1));
//                        camera.pitch(Deg(delta.1 as f32 * 0.1));
//                    },
//                    _ => {}
//                }
//                _ => {},
//            });
//        }
//
//        let cube_uniform_buffer_object = UniformBufferObject {
//            model: Matrix4::from_translation(vec3(0.0, -5.0, -10.0)) * Matrix4::from_scale(1.0),
//            view: camera.view_matrix(),
//            proj: perspective
//        };
//
//        let cube_uniform_buffer = CpuAccessibleBuffer::from_iter(
//            device.clone(), BufferUsage::all(), vec![cube_uniform_buffer_object].into_iter()
//        ).unwrap();
//
//        let cube_set = Arc::new(PersistentDescriptorSet::start(cube_pipeline.clone(), 0)
//            .add_buffer(cube_uniform_buffer.clone()).unwrap()
//            .build().unwrap()
//        );
//
//        let teapot_uniform_buffer_object = UniformBufferObject {
//            model: Matrix4::from_translation(vec3(0.0, 0.0, -10.0)) * Matrix4::from_scale(2.0),
//            view: camera.view_matrix(),
//            proj: perspective
//        };
//
//        let teapot_uniform_buffer = CpuAccessibleBuffer::from_iter(
//            device.clone(), BufferUsage::all(), vec![teapot_uniform_buffer_object].into_iter()
//        ).unwrap();
//
//        let camera_pos = camera.position().clone();
//
//        let camera_uniform_buffer = CpuAccessibleBuffer::from_iter(
//            device.clone(), BufferUsage::all(), vec![camera_pos].into_iter()
//        ).unwrap();
//
//        let teapot_set = Arc::new(PersistentDescriptorSet::start(teapot_pipeline.clone(), 0)
//            .add_buffer(teapot_uniform_buffer.clone()).unwrap()
//            .add_buffer(cube_uniform_buffer.clone()).unwrap()
//            .add_buffer(camera_uniform_buffer.clone()).unwrap()
//            .build().unwrap()
//        );
//
//        let (image_num, acquire_future) = match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
//            Ok(r) => r,
//            Err(e) => panic!("{:?}", e),
//        };
//
//        let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()];
//
//        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
//            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
//            .unwrap()
//            .draw_drawable(&teapot, &dynamic_state, (teapot_set).clone())
//            .unwrap()
//            .draw_drawable(&cube, &dynamic_state, (cube_set).clone())
//            .unwrap()
//            .end_render_pass()
//            .unwrap()
//            .build().unwrap();
//
//        let future = previous_frame_end.join(acquire_future)
//            .then_execute(queue.clone(), command_buffer)
//            .unwrap()
//            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
//            .then_signal_fence_and_flush();
//
//        match future {
//            Ok(f) => previous_frame_end = Box::new(f) as Box<_>,
//            Err(e) => {println!("{:?}", e); previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;},
//        }
//
//    }

}

fn window_size_dependent_setup(
    device: Arc<Device>,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let depth_buffer = AttachmentImage::transient(device.clone(), dimensions.width_height(), Format::D16Unorm).unwrap();

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
                .add(depth_buffer.clone()).unwrap()
                .build().unwrap()
        ) as Arc<FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}
