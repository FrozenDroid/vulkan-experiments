use specs::{System, ReadExpect, WriteExpect, ReadStorage, Write, Resources, Entities, WriteStorage};
use vulkano::device::{Device, Queue};
use vulkano::sync::GpuFuture;
use std::sync::Arc;
use vulkano::swapchain::Swapchain;
use vulkano::command_buffer::{DynamicState, AutoCommandBufferBuilder};
use std::time::Duration;
use vulkano::framebuffer::FramebufferAbstract;
use frozengame::model::Vertex;
use std::ops::DerefMut;
use specs::join::Join;
use crate::components::graphics::{Texture, MeshBuffer, GraphicsPipeline, FixedSizeDescriptorSetsPool, DescriptorSetsCollection};
use vulkano::format::{Format, R8G8B8A8Uint};
use crate::{UniformBufferObject};
use vulkano::buffer::CpuBufferPool;
use crate::components::player::ActivePlayer;
use crate::components::movement::Isometry;
use crate::components::physics::RigidBody;
use nphysics3d::world::World;
use vulkano::descriptor::DescriptorSet;
use itertools::Itertools;
use nalgebra::Vector3;

pub struct MeshUniformSystem;

impl<'a> System<'a> for MeshUniformSystem {
    type SystemData = (
        ReadStorage<'a, Isometry<f32>>,
        ReadStorage<'a, RigidBody>,
        ReadExpect<'a, World<f32>>,
        ReadStorage<'a, GraphicsPipeline>,
        ReadExpect<'a, ActivePlayer>,
        WriteExpect<'a, CpuBufferPool<crate::teapot_vs::ty::UniformBufferObject>>,
        WriteExpect<'a, CpuBufferPool<crate::teapot_fs::ty::LightObject>>,
        ReadExpect<'a, DynamicState>,
        WriteStorage<'a, FixedSizeDescriptorSetsPool>,
        WriteStorage<'a, DescriptorSetsCollection>,
        WriteExpect<'a, Arc<Device>>,
        Entities<'a>,
    );

    fn run(&mut self, (isometries, rigid_bodies, physics_world, pipelines, player, ubo_buffer_pool, light_buffer_pool, dynamic_state, mut descriptor_set_pools, mut descriptor_set_collection, _device, entities): Self::SystemData) {
        let mut rigid_body = rigid_bodies.get(player.0).unwrap();
        let mut rigid_body = physics_world.rigid_body(rigid_body.0).unwrap();

        let viewport = dynamic_state.clone().viewports.unwrap().first().unwrap().clone();

        for (entity, isometry, _pipeline, mut descriptor_set_pool, mut descriptor_set_collection) in (&entities, &isometries, &pipelines, &mut descriptor_set_pools, &mut descriptor_set_collection).join() {
            let mut perspective = nalgebra::Perspective3::new(
                viewport.dimensions[0] as f32 / viewport.dimensions[1] as f32,
                -(3.14 / 2.0),
                0.01,
                100.0
            );

            descriptor_set_collection.push_or_replace(
                1,
                Arc::new(
                    descriptor_set_pool.0.next()
                        .add_buffer(ubo_buffer_pool.next(crate::teapot_vs::ty::UniformBufferObject {
                            _dummy0: [0u8; 4],
                            cam_pos: rigid_body.position().translation.vector.into(),
                            model: isometry.0.to_homogeneous().into(),
                            view: rigid_body.position().to_homogeneous().into(),
                            proj: perspective.to_homogeneous().into(),
                        }).unwrap()).unwrap()
                        .enter_array().unwrap()
                        .add_buffer(light_buffer_pool.next(
                            crate::teapot_fs::ty::LightObject {
                                _dummy0: [0u8; 4],
                                position: [5.0, 10.0, 10.0],
                                color: [155.0, 155.0, 155.0],
                        }).unwrap()).unwrap()
                        .leave_array().unwrap()
                        .build().unwrap()
                )
            );
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
        ReadStorage<'a, DescriptorSetsCollection>,
        ReadStorage<'a, GraphicsPipeline>,
        WriteExpect<'a, Arc<Device>>,
        Write<'a, Option<Arc<Queue>>>,
        Write<'a, Vec<Arc<FramebufferAbstract + Send + Sync + 'static>>>,
        Write<'a, Option<Arc<Swapchain<winit::Window>>>>,
        Write<'a, DynamicState>,
        Write<'a, Duration>,
    );

    fn run(&mut self, (meshes, desciptor_set_collections, pipelines, mut device, mut queue, framebuffers, mut swapchain, dynamic_state, _delta): Self::SystemData) {
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

        command_buffer_builder = command_buffer_builder.begin_render_pass(framebuffers[image_num].clone(), false, clear_values).unwrap();

        for (mesh, desciptor_set_collection, pipeline) in (&meshes, &desciptor_set_collections, &pipelines).join() {
            command_buffer_builder = command_buffer_builder.draw_indexed(
                pipeline.0.clone(),
                &dynamic_state,
                mesh.0.clone(),
                mesh.1.clone(),
                desciptor_set_collection.0.clone(),
                ()
            ).unwrap();
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

