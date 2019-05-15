#![feature(crate_visibility_modifier)]
#![feature(vec_remove_item)]
#![feature(duration_float)]

use vulkano::device::{Device, Queue};
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage, CpuBufferPool, ImmutableBuffer};
use winit::{Window, VirtualKeyCode, ElementState, EventsLoop, ControlFlow, Event};
use std::sync::{Arc, RwLock};
use vulkano::framebuffer::{Framebuffer, Subpass, RenderPassAbstract, FramebufferAbstract};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::command_buffer::{DynamicState};
use vulkano::image::{SwapchainImage, AttachmentImage, ImmutableImage, Dimensions, ImageUsage, ImageLayout, MipmapsCount};
use vulkano::pipeline::viewport::Viewport;
use vulkano::descriptor::descriptor_set::{DescriptorSetsCollection, FixedSizeDescriptorSetsPool, PersistentDescriptorSet, PersistentDescriptorSetBuilder};
use std::path::Path;
use vulkano::image::traits::ImageViewAccess;
use crate::camera::Camera;
use vulkano::format::{Format, AcceptsPixels};
use frozengame::{FrozenGameBuilder};
use fuji::{FujiBuilder};
use frozengame::model::{Mesh};
use specs::prelude::*;
use crate::components::player::{ActivePlayer};
use nphysics3d::world::World as PhysicsWorld;
use tobj::load_obj;
use vulkano::swapchain::{Swapchain, Surface};
use std::time::{Duration};
use crate::systems::physics::PhysicsSystem;
use nphysics3d::object::{RigidBodyDesc, ColliderDesc};
use ncollide3d::shape::{Cuboid, ShapeHandle};
use crate::components::physics::RigidBody;
use nphysics3d::algebra::Velocity3;
use nalgebra::{Rotation3, Matrix4, Isometry3, Vector3};
use alga::general::RealField;
use crate::components::movement::Isometry;
use crate::systems::controls::EventSystem;
use nphysics3d::material::{MaterialHandle, BasicMaterial};
use crate::systems::graphics::{RenderSystem, MeshUniformSystem};
use vulkano::sampler::{MipmapMode, Sampler, Filter, SamplerAddressMode};
use gltf::Semantic;
use crate::components::graphics::{Texture, MeshBuffer};
use std::thread::sleep;
use std::io::Read;
use itertools::Itertools;

#[macro_use]
extern crate itertools;

mod systems;
mod camera;
mod components;

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


#[derive(Clone)]
pub struct UniformBufferObject<N: RealField> {
    cam_pos:    Vector3<N>,
    model:      Matrix4<N>,
    view:       Matrix4<N>,
    proj:       Matrix4<N>,
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
    world.register::<Isometry<f32>>();
    world.register::<RigidBody>();
    world.register::<components::graphics::GraphicsPipeline>();
    world.register::<components::graphics::MeshBuffer<Vertex, u32>>();
    world.register::<components::graphics::FixedSizeDescriptorSetsPool>();
    world.register::<components::graphics::DescriptorSetsCollection>();
    world.register::<Texture>();

    world.add_resource(Duration::from_secs(0));

    let engine_instance = FrozenGameBuilder::new(fuji).build();

    let fuji = Box::leak(Box::new(engine_instance.fuji.clone()));

    let surface = fuji.surface();
    let swapchain = fuji.swapchain();
    let swapchain_images = fuji.swapchain_images();
    let events_loop: &RwLock<EventsLoop> = fuji.events_loop();
    let queue: &Arc<Queue> = fuji.graphics_queue();
    let device = fuji.device();

    world.add_resource(CpuBufferPool::new(device.clone(), BufferUsage::all()) as CpuBufferPool<crate::teapot_vs::ty::UniformBufferObject>);
    world.add_resource(CpuBufferPool::new(device.clone(), BufferUsage::all()) as CpuBufferPool<crate::teapot_fs::ty::LightObject>);

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

    let _cube_vs = cube_vs::Shader::load(device.clone()).unwrap();
    let _cube_fs = cube_fs::Shader::load(device.clone()).unwrap();

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
        .blend_alpha_blending()
        .fragment_shader(teapot_fs.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
    );

    let sampler = Sampler::new(
        device.clone(),
        Filter::Linear,
        Filter::Linear,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0, 1.0, 0.0, 0.0
    ).unwrap();

    let (_document, _buffers, images) = gltf::import("9_mm/scene.gltf").unwrap();
    for mesh in _document.meshes() {
        if mesh.name() == Some("Plane001_Plane_0") {
            break;
        }
        dbg!(mesh.name());

        for primitive in mesh.primitives() {
            let mut normal_tex_coords = vec![];
            let reader = primitive.reader(|buffer| Some(&_buffers[buffer.index()]));

            let mut descriptor_sets_collection = components::graphics::DescriptorSetsCollection::default();

            if let (
                Some(color), Some(roughness), Some(emissive), Some(occlusion), Some(normal)
            ) = (primitive.material().pbr_metallic_roughness().base_color_texture(),
                 primitive.material().pbr_metallic_roughness().metallic_roughness_texture(),
                 primitive.material().emissive_texture(),
                 primitive.material().occlusion_texture(),
                 primitive.material().normal_texture())
            {

                match reader.read_tex_coords(0) {
                    Some(gltf::mesh::util::ReadTexCoords::F32(t)) => normal_tex_coords.extend(t),
                    _ => {}
                }

                let mut color_texture = &images[color.texture().index()];

                dbg!(color_texture.format);
                use vulkano::sync::GpuFuture;
                let color_image = ImmutableImage::from_iter(
                    color_texture.pixels.clone().into_iter(), Dimensions::Dim2d { width: color_texture.width, height: color_texture.height }, Format::R8G8B8A8Srgb, queue.clone()
                ).unwrap();

                (color_image.1).then_signal_fence_and_flush().unwrap().wait(None).unwrap();

                let roughness_texture = &images[roughness.texture().index()];
                dbg!(roughness_texture.format);

                let pixels: Vec<u8> = roughness_texture.pixels.clone().into_iter().tuples().map(|(a, b, c)| vec![a, b, c, std::u8::MAX]).flatten().collect();

                let roughness_image = ImmutableImage::from_iter(
                    pixels.into_iter(), Dimensions::Dim2d { width: roughness_texture.width, height: roughness_texture.height }, Format::R8G8B8A8Srgb, queue.clone()
                ).unwrap();

                let emissive_texture = &images[emissive.texture().index()];
                dbg!(emissive_texture.format);

                let pixels: Vec<u8> = emissive_texture.pixels.clone().into_iter().tuples().map(|(a, b, c)| vec![a, b, c, std::u8::MAX]).flatten().collect();

                let emissive_image = ImmutableImage::from_iter(
                    pixels.into_iter(), Dimensions::Dim2d { width: emissive_texture.width, height: emissive_texture.height }, Format::R8G8B8A8Srgb, queue.clone()
                ).unwrap();

                let occlusion_texture = &images[occlusion.texture().index()];
                dbg!(occlusion_texture.format);

                let pixels: Vec<u8> = occlusion_texture.pixels.clone().into_iter().tuples().map(|(a, b, c)| vec![a, b, c, std::u8::MAX]).flatten().collect();

                let occlusion_image = ImmutableImage::from_iter(
                    pixels.into_iter(), Dimensions::Dim2d { width: occlusion_texture.width, height: occlusion_texture.height }, Format::R8G8B8A8Srgb, queue.clone()
                ).unwrap();

                let normal_texture = &images[normal.texture().index()];
                dbg!(normal_texture.format);

                let normal_image = ImmutableImage::from_iter(
                    normal_texture.pixels.clone().into_iter(), Dimensions::Dim2d { width: normal_texture.width, height: normal_texture.height }, Format::R8G8B8A8Srgb, queue.clone()
                ).unwrap();

                let texture_descriptor = PersistentDescriptorSet::start(teapot_pipeline.clone(), 0)
                    .add_sampled_image(color_image.0.clone(), sampler.clone()).unwrap()
                    .add_sampled_image(roughness_image.0.clone(), sampler.clone()).unwrap()
                    .add_sampled_image(emissive_image.0.clone(), sampler.clone()).unwrap()
                    .add_sampled_image(occlusion_image.0.clone(), sampler.clone()).unwrap()
                    .add_sampled_image(normal_image.0.clone(), sampler.clone()).unwrap()
                    .build().unwrap();

                descriptor_sets_collection.push_or_replace(0, Arc::new(texture_descriptor));
            } else {
                println!("WARNING! NO TEXTURE");
            }

            let mut vertices: Vec<Vertex> = vec![];

            for (position, normals, tex_coords) in itertools::izip!(reader.read_positions().unwrap(), reader.read_normals().unwrap(), normal_tex_coords) {
                vertices.push(Vertex {
                    position,
                    normals,
                    tex_coords
                });
            }

            let mesh: Mesh<Vertex, u32> = Mesh {
                indices: match reader.read_indices().unwrap() {
                    gltf::mesh::util::ReadIndices::U32(iter) => iter.collect(),
                    _ => vec![]
                },
                vertices,
            };

            let immutable_indices_buf = CpuAccessibleBuffer::from_iter(
                device.clone(), BufferUsage::all(), mesh.indices.into_iter()
            ).unwrap();

            let immutable_vert_buf = CpuAccessibleBuffer::from_iter(
                device.clone(), BufferUsage::all(), mesh.vertices.into_iter(),
            ).unwrap();

            world.create_entity()
                .with(MeshBuffer::<Vertex, u32>::from(vec![immutable_vert_buf.clone()], immutable_indices_buf.clone()))
                .with(components::graphics::GraphicsPipeline(teapot_pipeline.clone()))
                .with(components::graphics::FixedSizeDescriptorSetsPool(FixedSizeDescriptorSetsPool::new(teapot_pipeline.clone(), 1)))
                .with(Isometry::<f32>::default())
                .with(descriptor_sets_collection)
                .build();

//            break;
        }
//        break;
    }

    let (models, _materials) = load_obj(&Path::new("./teapot.obj")).unwrap();

    use frozengame::model::Vertex;

    let mut physics_world: PhysicsWorld<f32> = PhysicsWorld::new();

    let shape = ShapeHandle::new(Cuboid::new(nalgebra::Vector3::new(0.5, 1.0, 0.5)));

    let collider = ColliderDesc::new(shape)
        .density(1.3)
        .material(MaterialHandle::new(BasicMaterial::new(1.2, 0.8)))
        .margin(0.02);

    let rigid_body = RigidBodyDesc::new()
        .name("player body".to_owned())
        .collider(&collider)
        .mass(2.2)
        .velocity(Velocity3::linear(0.0, 0.0, 0.0))
        .build(&mut physics_world);

    let self_player = world.create_entity().with(RigidBody(rigid_body.handle())).build();
    world.add_resource(ActivePlayer(self_player));

    let mut dynamic_state = DynamicState::default();

    let framebuffers = window_size_dependent_setup(device.clone(), &swapchain_images, render_pass.clone(), &mut dynamic_state);

    world.add_resource(physics_world);
    world.add_resource::<Option<Arc<Swapchain<winit::Window>>>>(Some(swapchain.clone()));
    world.add_resource::<Vec<Arc<FramebufferAbstract + Send + Sync>>>(framebuffers);
    world.add_resource::<Arc<Device>>(device.clone());
    world.add_resource::<Option<Arc<Queue>>>(Some(queue.clone()));

    let (image, init) = ImmutableImage::uninitialized(
        device.clone(),
        Dimensions::Dim2d { width: 200, height: 200 },
        swapchain.format(),
        MipmapsCount::One,
        ImageUsage { sampled: true, ..ImageUsage::none() },
        ImageLayout::General,
        device.active_queue_families()
    ).unwrap();

    let mut dispatcher = DispatcherBuilder::new()
        .with(PhysicsSystem::<f32>::default(), "physics system", &[])
        .with(EventSystem, "event system", &[])
        .with(MeshUniformSystem, "mesh system", &[])
        .with(RenderSystem { previous_frame: None }, "render system", &["mesh system"])
        .build();

    world.add_resource(dynamic_state);
    dispatcher.setup(&mut world.res);
    dispatcher.dispatch(&mut world.res);
    world.maintain();

    let mut events = Vec::new();
    world.add_resource(events.clone());

    loop {
        events.clear();
        if let Ok(ref mut events_loop) = events_loop.write() {
            events_loop.poll_events(|event| events.push(event));
        }

        world.add_resource(events.clone());

        dispatcher.dispatch(&world.res);
    }

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
