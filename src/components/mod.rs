use cgmath::{Vector3, Matrix4, Rad, Deg};
use specs::{Component, VecStorage, System, ReadStorage, Read, WriteStorage, Write, Resources, WriteExpect, ReadExpect, Entities, RunNow};
use frozengame::model::{Vertex};
use vulkano::pipeline::input_assembly::Index;
use std::sync::{Arc, RwLock, Mutex, RwLockWriteGuard};
use vulkano::buffer::{TypedBufferAccess, BufferAccess, CpuAccessibleBuffer, CpuBufferPool};
use vulkano::pipeline::{GraphicsPipelineAbstract};
use vulkano::pipeline::vertex::VertexSource;
use std::marker::PhantomData;
use specs::join::Join;
use winit::{VirtualKeyCode, ElementState, Event};
use core::borrow::BorrowMut;
use crate::camera::Camera;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, Queue};
use vulkano::instance::QueueFamily;
use shred::{SetupHandler, Resource};
use std::ops::DerefMut;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::swapchain::Swapchain;
use crate::UniformBufferObject;
use crate::components::player::ActivePlayer;
use vulkano::descriptor::{DescriptorSet, PipelineLayoutAbstract};
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, FixedSizeDescriptorSetsPool};
use vulkano::sync::{GpuFuture, NowFuture};
use std::cell::RefCell;
use std::time::{Duration, Instant};

pub mod player;

#[derive(Debug, Clone, Copy)]
pub struct Position(Vector3<f32>);

impl Default for Position {
    fn default() -> Self {
        Position(Vector3::new(0.0, 0.0, 0.0))
    }
}

pub struct MeshBuffer<VD, ID>(Vec<Arc<BufferAccess + Send + Sync + 'static>>, Arc<TypedBufferAccess<Content = [ID]> + Send + Sync + 'static>, PhantomData<VD>);

impl<VD, ID> MeshBuffer<VD, ID> {
    pub fn from(v: Vec<Arc<BufferAccess + Send + Sync + 'static>>, i: Arc<TypedBufferAccess<Content = [ID]> + Send + Sync + 'static>) -> Self {
        MeshBuffer(v, i, PhantomData::default())
    }
}

impl<VD, ID> Component for MeshBuffer<VD, ID>
    where
        MeshBuffer<VD, ID>: Send + Sync + 'static
{
    type Storage = VecStorage<Self>;
}

impl Component for Position {
    type Storage = VecStorage<Self>;
}

pub struct Mesh<VD, IT>(pub frozengame::model::Mesh<VD, IT>);

impl Component for Mesh<Vertex, u32> {
    type Storage = VecStorage<Self>;
}

#[derive(Default)]
pub struct PressedKeys(pub Vec<VirtualKeyCode>);

pub struct MovementSystem;

pub struct EventSystem;
use std::ops::Deref;
impl<'a> System<'a> for EventSystem {
    type SystemData = Read<'a, Option<Event>>;

    fn run(&mut self, event: Self::SystemData) {
//        match event.deref() {
//            Some(ev) => {
//                ev
//            },
//            _ => {}
//        }
    }
}

impl<'a> System<'a> for MovementSystem {
    type SystemData = (
        Read<'a, PressedKeys>,
        ReadExpect<'a, ActivePlayer>,
        WriteStorage<'a, Position>,
        WriteStorage<'a, Camera<f32>>,
        ReadExpect<'a, Duration>,
    );

    fn run(&mut self, (pressed_keys, player, mut positions, mut camera, duration): Self::SystemData) {
        let position = match positions.get_mut(player.0) {
            Some(p) => p.clone(),
            _ => Position::default(),
        };

        let mut camera = camera.get_mut(player.0).expect("couldn't get camera");

        for pressed_key in pressed_keys.0.iter() {
//            println!("pressed key {:?}", pressed_key);
            match pressed_key {
                VirtualKeyCode::W => camera.move_forward(0.01 as f32),
                VirtualKeyCode::S => camera.move_forward(-0.01 as f32),
                VirtualKeyCode::A => camera.move_left(0.01 as f32),
                VirtualKeyCode::D => camera.move_left(-0.01 as f32),
                _ => {},
            };
        }

        positions.insert(player.0, Position(camera.position().clone() as Vector3<f32>));
    }
}

impl Component for Camera<f32> {
    type Storage = VecStorage<Self>;
}

impl<'a> System<'a> for Camera<f32> {
    type SystemData = (
        ReadStorage<'a, Position>,
        WriteStorage<'a, Camera<f32>>,
        ReadExpect<'a, ActivePlayer>,
    );

    fn run(&mut self, (positions, mut cameras, mut player): Self::SystemData) {
        let mut camera = cameras.get_mut(player.0).expect("couldn't get player camera");
        if let Some(p) = positions.get(player.0) {
//            camera.set_position(p.0.clone());
        }
        println!("updating camera system");

    }
}

pub struct GraphicsPipeline(pub Arc<GraphicsPipelineAbstract + Send + Sync + 'static>);

impl Component for GraphicsPipeline {
    type Storage = VecStorage<Self>;
}

pub struct DescriptorSetPool(pub FixedSizeDescriptorSetsPool<Arc<PipelineLayoutAbstract + Send + Sync>>);

impl Component for DescriptorSetPool
{
    type Storage = VecStorage<Self>;
}

pub struct MeshDescriptorSet(pub Arc<DescriptorSet + Send + Sync>);

impl Component for MeshDescriptorSet {
    type Storage = VecStorage<Self>;
}

pub struct MeshUniformSystem;

impl<'a> System<'a> for MeshUniformSystem {
    type SystemData = (
        ReadStorage<'a, Position>,
        ReadStorage<'a, Camera<f32>>,
        ReadStorage<'a, GraphicsPipeline>,
        WriteStorage<'a, MeshDescriptorSet>,
        ReadExpect<'a, ActivePlayer>,
        WriteExpect<'a, CpuBufferPool<UniformBufferObject>>,
        ReadExpect<'a, DynamicState>,
        WriteStorage<'a, DescriptorSetPool>,
        WriteExpect<'a, Arc<Device>>,
        Entities<'a>,
    );

    fn run(&mut self, (positions, cameras, pipelines, mut mesh_descriptor_sets, player, buffer_pool, dynamic_state, mut descriptor_set_pools, mut device, entities): Self::SystemData) {
        let camera = cameras.get(player.0).expect("couldn't get player's camera");

        let viewport = dynamic_state.clone().viewports.unwrap().first().unwrap().clone();

        for (entity, position, pipeline, descriptor_set_pool) in (&entities, &positions, &pipelines, &mut descriptor_set_pools).join() {
            let mut perspective = cgmath::perspective(
                Rad::from(Deg(45.0)),
                viewport.dimensions[0] as f32 / viewport.dimensions[1] as f32,
                0.01,
                100.0,
            );
            perspective.y.y *= -1.0;
            mesh_descriptor_sets.insert(entity, MeshDescriptorSet(Arc::new((descriptor_set_pool.0).next().add_buffer(buffer_pool.next(UniformBufferObject {
                model: Matrix4::from_translation(position.0),
                view: camera.view_matrix(),
                proj: perspective,
            }).unwrap()).unwrap().build().unwrap())));
        }
    }

}

pub struct RenderSystem {
    pub previous_frame: Option<Box<GpuFuture + Send + Sync>>
}

impl<'a> System<'a> for RenderSystem
{
    type SystemData = (
        ReadStorage<'a, MeshBuffer<Vertex, u32>>,
        ReadStorage<'a, MeshDescriptorSet>,
        ReadStorage<'a, GraphicsPipeline>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, Camera<f32>>,
        WriteExpect<'a, Arc<Device>>,
        Write<'a, Option<Arc<Queue>>>,
        Write<'a, Vec<Arc<FramebufferAbstract + Send + Sync + 'static>>>,
        Write<'a, Option<Arc<Swapchain<winit::Window>>>>,
        Write<'a, DynamicState>,
        Write<'a, Duration>,
    );

    fn run(&mut self, (meshes, mut desciptor_sets, mut pipelines, pos, camera, mut device, mut queue, mut framebuffers, mut swapchain, mut dynamic_state, mut delta): Self::SystemData) {
        if let Some(ref mut f) = self.previous_frame {
            f.cleanup_finished();
        }
        let device = device.deref_mut().clone();
        let queue  = queue.deref_mut().clone().expect("queue not present");
        let swapchain = swapchain.deref_mut().clone().expect("swapchain not present");

        let (image_num, acquire_future) = match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(e) => panic!("{:?}", e),
        };

        // Render all meshes
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(
            device.clone(),
            queue.clone().family(),
        ).unwrap();

        let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()];

        use vulkano::SafeDeref;

        command_buffer_builder = command_buffer_builder.begin_render_pass(framebuffers[image_num].clone(), false, clear_values).unwrap();
        for (mesh, desciptor_sets, pipeline, pos) in (&meshes, &desciptor_sets, &pipelines, &pos).join() {
            command_buffer_builder = command_buffer_builder.draw_indexed(pipeline.0.clone(), &dynamic_state, mesh.0.clone(), mesh.1.clone(), (desciptor_sets.0.clone()), ()).unwrap();
            println!("Rendering model");
        }
        command_buffer_builder = command_buffer_builder.end_render_pass().unwrap();

        let future = self.previous_frame.take().unwrap().join(acquire_future)
            .then_execute(queue.clone(), command_buffer_builder.build().unwrap()).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(f) => {
                self.previous_frame = Some(Box::new(f) as Box<_>);
            },
            Err(e) => {println!("{:?}", e); self.previous_frame = Some(Box::new(vulkano::sync::now(device.clone())) as Box<_>);},
        };
    }

    fn setup(&mut self, res: &mut Resources) {
        let device = res.fetch::<Arc<Device>>().clone();

        self.previous_frame = Some(Box::new(vulkano::sync::now(device)));
    }
}
