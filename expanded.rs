#![feature(prelude_import)]
#![no_std]
#![feature(crate_visibility_modifier)]
#![feature(vec_remove_item)]
#![feature(duration_float)]
#[prelude_import]
use ::std::prelude::v1::*;
#[macro_use]
extern crate std as std;

use vulkano::device::{Device, Queue};
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage, CpuBufferPool,
                      ImmutableBuffer};
use winit::{Window, VirtualKeyCode, ElementState, EventsLoop, ControlFlow,
            Event};
use std::sync::{Arc, RwLock};
use vulkano::framebuffer::{Framebuffer, Subpass, RenderPassAbstract,
                           FramebufferAbstract};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::command_buffer::{DynamicState};
use vulkano::image::{SwapchainImage, AttachmentImage, ImmutableImage,
                     Dimensions, ImageUsage, ImageLayout, MipmapsCount};
use vulkano::pipeline::viewport::Viewport;
use vulkano::descriptor::descriptor_set::{DescriptorSetsCollection,
                                          FixedSizeDescriptorSetsPool,
                                          PersistentDescriptorSet,
                                          PersistentDescriptorSetBuilder};
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

mod systems {


























    //                let pool = CpuBufferPool::new(device.clone(), BufferUsage::all()) as CpuBufferPool<[u8; 4]>;


    //                ImmutableImage::from_buffer(
    //                    pool.next(color_texture.pixels.clone().into_iter().collect()).unwrap(),
    //                    Dimensions::Dim2d { width: color_texture.width, height: color_texture.height },
    //                    Format::R8G8B8A8Srgb,
    //                    queue.clone()
    //                );






















    //            break;
    //        break;






















    pub mod controls {
        use specs::{System, Read, ReadExpect, ReadStorage, WriteExpect,
                    WriteStorage};
        use nphysics3d::world::World;
        use winit::{Event, KeyboardInput, VirtualKeyCode, WindowEvent};
        use std::ops::Deref;
        use crate::components::player::ActivePlayer;
        use crate::components::movement::Isometry;
        use nalgebra::{Isometry3, Vector3, Vector, UnitQuaternion, Unit};
        use crate::components::physics::RigidBody;
        use nphysics3d::algebra::{Velocity3, Force3, ForceType};
        use nphysics3d::object::Body;
        use std::sync::RwLock;
        pub struct EventSystem;
        impl <'a> System<'a> for EventSystem {
            type
            SystemData
            =
            (Read<'a, Vec<Event>>, ReadExpect<'a, ActivePlayer>,
             WriteStorage<'a, RigidBody>, WriteExpect<'a, World<f32>>);
            fn run(&mut self,
                   (events, active_player, mut rigid_bodies,
                    mut physics_world): Self::SystemData) {
                let mut rigid_body =
                    rigid_bodies.get_mut(active_player.0).expect("expected player isometry");
                let mut rigid_body =
                    physics_world.rigid_body_mut(rigid_body.0.clone()).unwrap();
                for event in events.clone().into_iter() {
                    match event {
                        Event::WindowEvent { event, .. } => {
                            match event {
                                WindowEvent::KeyboardInput { input, .. } => {
                                    match input.virtual_keycode {
                                        Some(VirtualKeyCode::W) => {
                                            rigid_body.apply_force(0,
                                                                   &Force3::linear(Vector3::new(0.0,
                                                                                                0.0,
                                                                                                100.0)),
                                                                   ForceType::Force,
                                                                   true);
                                        }
                                        Some(VirtualKeyCode::S) => {
                                            rigid_body.apply_force(0,
                                                                   &Force3::linear(Vector3::new(0.0,
                                                                                                0.0,
                                                                                                -100.0)),
                                                                   ForceType::Force,
                                                                   true);
                                        }
                                        Some(VirtualKeyCode::A) => {
                                            rigid_body.apply_force(0,
                                                                   &Force3::linear(Vector3::new(-100.0,
                                                                                                0.0,
                                                                                                0.0)),
                                                                   ForceType::Force,
                                                                   true);
                                        }
                                        Some(VirtualKeyCode::D) => {
                                            rigid_body.apply_force(0,
                                                                   &Force3::linear(Vector3::new(100.0,
                                                                                                0.0,
                                                                                                0.0)),
                                                                   ForceType::Force,
                                                                   true);
                                        }
                                        _ => { }
                                    };
                                }
                                _ => { }
                            };
                        }
                        Event::DeviceEvent { event, .. } => {
                            match event {
                                winit::DeviceEvent::MouseMotion { delta } => {
                                    rigid_body.set_position(UnitQuaternion::from_axis_angle(&(rigid_body.position().rotation
                                                                                                  *
                                                                                                  Vector3::y_axis()),
                                                                                            (-delta.0
                                                                                                 as
                                                                                                 f32)
                                                                                                *
                                                                                                0.01)
                                                                *
                                                                rigid_body.position());
                                    rigid_body.set_position(UnitQuaternion::from_axis_angle(&(Vector3::x_axis()),
                                                                                            (delta.1
                                                                                                 as
                                                                                                 f32)
                                                                                                *
                                                                                                0.01)
                                                                *
                                                                rigid_body.position());
                                }
                                _ => { }
                            }
                        }
                        _ => { }
                    };
                }
            }
        }
    }
    pub mod physics {
        use specs::{System, WriteExpect, WriteStorage};
        use alga::general::RealField;
        use std::marker::PhantomData;
        use nphysics3d::world::World as PhysicsWorld;
        use crate::components::physics::{Shape, RigidBody};
        use specs::join::Join;
        use std::time::{Duration, Instant};
        pub struct PhysicsSystem<N>(PhantomData<N>, pub Instant);
        impl <N> Default for PhysicsSystem<N> {
            fn default() -> Self {
                PhysicsSystem(PhantomData::default(), Instant::now())
            }
        }
        impl <'a, N: RealField> System<'a> for PhysicsSystem<N> {
            type
            SystemData
            =
            (WriteExpect<'a, PhysicsWorld<N>>, WriteStorage<'a, RigidBody>);
            fn run(&mut self,
                   (mut physics_world, rigid_bodies): Self::SystemData) {
                if Instant::now().duration_since(self.1).as_secs_f64() >=
                       physics_world.timestep().to_subset().unwrap() {
                    physics_world.step();
                    self.1 = Instant::now();
                }
            }
        }
    }
    pub mod graphics {
        use specs::{System, ReadExpect, WriteExpect, ReadStorage, Write,
                    Resources, Entities, WriteStorage};
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
        use crate::components::graphics::{Texture, MeshBuffer,
                                          GraphicsPipeline,
                                          FixedSizeDescriptorSetsPool,
                                          DescriptorSetsCollection};
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
        impl <'a> System<'a> for MeshUniformSystem {
            type
            SystemData
            =
            (ReadStorage<'a, Isometry<f32>>, ReadStorage<'a, RigidBody>,
             ReadExpect<'a, World<f32>>, ReadStorage<'a, GraphicsPipeline>,
             ReadExpect<'a, ActivePlayer>,
             WriteExpect<'a, CpuBufferPool<UniformBufferObject<f32>>>,
             WriteExpect<'a,
                         CpuBufferPool<crate::teapot_fs::ty::LightObject>>,
             ReadExpect<'a, DynamicState>,
             WriteStorage<'a, FixedSizeDescriptorSetsPool>,
             WriteStorage<'a, DescriptorSetsCollection>,
             WriteExpect<'a, Arc<Device>>, Entities<'a>);
            fn run(&mut self,
                   (isometries, rigid_bodies, physics_world, pipelines,
                    player, ubo_buffer_pool, light_buffer_pool, dynamic_state,
                    mut descriptor_set_pools, mut descriptor_set_collection,
                    _device, entities): Self::SystemData) {
                let mut rigid_body = rigid_bodies.get(player.0).unwrap();
                let mut rigid_body =
                    physics_world.rigid_body(rigid_body.0).unwrap();
                let viewport =
                    dynamic_state.clone().viewports.unwrap().first().unwrap().clone();
                for (entity, isometry, _pipeline, mut descriptor_set_pool,
                     mut descriptor_set_collection) in
                    (&entities, &isometries, &pipelines,
                     &mut descriptor_set_pools,
                     &mut descriptor_set_collection).join() {
                    let mut perspective =
                        nalgebra::Perspective3::new(viewport.dimensions[0] as
                                                        f32 /
                                                        viewport.dimensions[1]
                                                            as f32,
                                                    -(3.14 / 2.0), 0.01,
                                                    100.0);
                    descriptor_set_collection.push_or_replace(1,
                                                              Arc::new(descriptor_set_pool.0.next().add_buffer(ubo_buffer_pool.next(UniformBufferObject{cam_pos:
                                                                                                                                                            rigid_body.position().translation.vector,
                                                                                                                                                        model:
                                                                                                                                                            isometry.0.to_homogeneous(),
                                                                                                                                                        view:
                                                                                                                                                            rigid_body.position().to_homogeneous(),
                                                                                                                                                        proj:
                                                                                                                                                            perspective.to_homogeneous(),}).unwrap()).unwrap().enter_array().unwrap().add_buffer(light_buffer_pool.next(crate::teapot_fs::ty::LightObject{_dummy0:
                                                                                                                                                                                                                                                                                                              [0u8;
                                                                                                                                                                                                                                                                                                                  4],
                                                                                                                                                                                                                                                                                                          position:
                                                                                                                                                                                                                                                                                                              [0.0,
                                                                                                                                                                                                                                                                                                               0.0,
                                                                                                                                                                                                                                                                                                               10.0],
                                                                                                                                                                                                                                                                                                          color:
                                                                                                                                                                                                                                                                                                              [155.0,
                                                                                                                                                                                                                                                                                                               155.0,
                                                                                                                                                                                                                                                                                                               155.0],}).unwrap()).unwrap().leave_array().unwrap().build().unwrap()));
                }
            }
        }
        pub struct RenderSystem {
            pub previous_frame: Option<Box<GpuFuture + Send + Sync>>,
        }
        impl <'a> System<'a> for RenderSystem {
            type
            SystemData
            =
            (ReadStorage<'a, MeshBuffer<Vertex, u32>>,
             ReadStorage<'a, DescriptorSetsCollection>,
             ReadStorage<'a, GraphicsPipeline>, WriteExpect<'a, Arc<Device>>,
             Write<'a, Option<Arc<Queue>>>,
             Write<'a, Vec<Arc<FramebufferAbstract + Send + Sync + 'static>>>,
             Write<'a, Option<Arc<Swapchain<winit::Window>>>>,
             Write<'a, DynamicState>, Write<'a, Duration>);
            fn run(&mut self,
                   (meshes, desciptor_set_collections, pipelines, mut device,
                    mut queue, framebuffers, mut swapchain, dynamic_state,
                    _delta): Self::SystemData) {
                if let Some(ref mut f) = self.previous_frame {
                    f.cleanup_finished();
                }
                let device = device.deref_mut().clone();
                let queue =
                    queue.deref_mut().clone().expect("queue not present");
                let swapchain =
                    swapchain.deref_mut().clone().expect("swapchain not present");
                let (image_num, acquire_future) =
                    match vulkano::swapchain::acquire_next_image(swapchain.clone(),
                                                                 None) {
                        Ok(r) => r,
                        Err(e) => {
                            ::std::rt::begin_panic_fmt(&::std::fmt::Arguments::new_v1(&[""],
                                                                                      &match (&e,)
                                                                                           {
                                                                                           (arg0,)
                                                                                           =>
                                                                                           [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                        ::std::fmt::Debug::fmt)],
                                                                                       }),
                                                       &("src/systems/graphics.rs",
                                                         109u32, 23u32))
                        }
                    };
                let mut command_buffer_builder =
                    AutoCommandBufferBuilder::primary_one_time_submit(device.clone(),
                                                                      queue.clone().family()).unwrap();
                let clear_values =
                    <[_]>::into_vec(box
                                        [[0.0, 0.0, 1.0, 1.0].into(),
                                         1f32.into()]);
                command_buffer_builder =
                    command_buffer_builder.begin_render_pass(framebuffers[image_num].clone(),
                                                             false,
                                                             clear_values).unwrap();
                for (mesh, desciptor_set_collection, pipeline) in
                    (&meshes, &desciptor_set_collections, &pipelines).join() {
                    command_buffer_builder =
                        command_buffer_builder.draw_indexed(pipeline.0.clone(),
                                                            &dynamic_state,
                                                            mesh.0.clone(),
                                                            mesh.1.clone(),
                                                            desciptor_set_collection.0.clone(),
                                                            ()).unwrap();
                }
                command_buffer_builder =
                    command_buffer_builder.end_render_pass().unwrap();
                let future =
                    self.previous_frame.take().unwrap().join(acquire_future).then_execute(queue.clone(),
                                                                                          command_buffer_builder.build().unwrap()).unwrap().then_swapchain_present(queue.clone(),
                                                                                                                                                                   swapchain.clone(),
                                                                                                                                                                   image_num).then_signal_fence_and_flush();
                match future {
                    Ok(f) => {
                        self.previous_frame = Some(Box::new(f) as Box<_>);
                    }
                    Err(e) => {
                        {
                            ::std::io::_print(::std::fmt::Arguments::new_v1(&["",
                                                                              "\n"],
                                                                            &match (&e,)
                                                                                 {
                                                                                 (arg0,)
                                                                                 =>
                                                                                 [::std::fmt::ArgumentV1::new(arg0,
                                                                                                              ::std::fmt::Debug::fmt)],
                                                                             }));
                        };
                        self.previous_frame =
                            Some(Box::new(vulkano::sync::now(device.clone()))
                                     as Box<_>);
                    }
                };
            }
            fn setup(&mut self, res: &mut Resources) {
                let device = res.fetch::<Arc<Device>>().clone();
                self.previous_frame =
                    Some(Box::new(vulkano::sync::now(device)));
            }
        }
    }
}
mod camera {
    use nalgebra::{Isometry3, Isometry};
    use alga::general::RealField;
    pub struct Camera<N: RealField> {
        isometry: Isometry3<N>,
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl <N: ::std::fmt::Debug + RealField> ::std::fmt::Debug for Camera<N> {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            match *self {
                Camera { isometry: ref __self_0_0 } => {
                    let mut debug_trait_builder = f.debug_struct("Camera");
                    let _ =
                        debug_trait_builder.field("isometry",
                                                  &&(*__self_0_0));
                    debug_trait_builder.finish()
                }
            }
        }
    }
    impl <T> Camera<T> where T: RealField { }
    impl <N: RealField> Default for Camera<N> {
        fn default() -> Self {
            Camera{isometry:
                       Isometry3::new(nalgebra::zero(), nalgebra::zero()),}
        }
    }
}
mod components {
    use specs::{Component, VecStorage, System, ReadStorage, Read,
                WriteStorage, Write, Resources, WriteExpect, ReadExpect,
                Entities};
    use frozengame::model::{Vertex};
    use std::sync::{Arc};
    use vulkano::buffer::{TypedBufferAccess, BufferAccess, CpuBufferPool};
    use vulkano::pipeline::{GraphicsPipelineAbstract};
    use std::marker::PhantomData;
    use specs::join::Join;
    use winit::{VirtualKeyCode, Event};
    use crate::camera::Camera;
    use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
    use vulkano::device::{Device, Queue};
    use std::ops::DerefMut;
    use vulkano::framebuffer::FramebufferAbstract;
    use vulkano::swapchain::Swapchain;
    use crate::UniformBufferObject;
    use crate::components::player::ActivePlayer;
    use vulkano::descriptor::{DescriptorSet, PipelineLayoutAbstract};
    use vulkano::descriptor::descriptor_set::{FixedSizeDescriptorSetsPool};
    use vulkano::sync::{GpuFuture};
    use std::time::{Duration};
    use crate::components::physics::RigidBody;
    use nphysics3d::world::World;
    use std::ops::Deref;
    use winit::WindowEvent::KeyboardInput;
    use nphysics3d::algebra::Velocity3;
    use nalgebra::{Point, Isometry3, Vector3, Matrix4, Point3};
    use alga::general::RealField;
    use crate::components::movement::Isometry;
    use vulkano::sampler::Sampler;
    use vulkano::image::ImmutableImage;
    use vulkano::format::FormatDesc;
    use crate::components::graphics::{GraphicsPipeline};
    pub mod player {
        use specs::{Entity, Component, VecStorage};
        use crate::components::movement::Isometry;
        use alga::general::RealField;
        pub struct ActivePlayer(pub Entity);
    }
    pub mod physics {
        use specs::{Component, VecStorage};
        use alga::general::RealField;
        use ncollide3d::shape::Shape as PhysicsShape;
        use nphysics3d::math::Force as PhysicsForce;
        use std::sync::Arc;
        use nphysics3d::object::{RigidBody as PhysicsRigidBody, BodyHandle};
        pub struct Shape<N: RealField>(Arc<PhysicsShape<N>>);
        impl <N: RealField> Component for Shape<N> {
            type
            Storage
            =
            VecStorage<Self>;
        }
        pub struct RigidBody(pub BodyHandle);
        impl Component for RigidBody {
            type
            Storage
            =
            VecStorage<Self>;
        }
        pub struct Force<N: RealField>(Arc<PhysicsForce<N>>);
        impl <N: RealField> Component for Force<N> {
            type
            Storage
            =
            VecStorage<Self>;
        }
    }
    pub mod movement {
        use nalgebra::{RealField, Isometry3};
        use specs::{Component, VecStorage};
        impl <N: RealField> From<Isometry3<N>> for Isometry<N> {
            fn from(isometry: Isometry3<N>) -> Self { Isometry(isometry) }
        }
        pub struct Isometry<N: RealField>(pub Isometry3<N>);
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl <N: ::std::fmt::Debug + RealField> ::std::fmt::Debug for
         Isometry<N> {
            fn fmt(&self, f: &mut ::std::fmt::Formatter)
             -> ::std::fmt::Result {
                match *self {
                    Isometry(ref __self_0_0) => {
                        let mut debug_trait_builder =
                            f.debug_tuple("Isometry");
                        let _ = debug_trait_builder.field(&&(*__self_0_0));
                        debug_trait_builder.finish()
                    }
                }
            }
        }
        impl <N: RealField> Default for Isometry<N> {
            fn default() -> Self {
                Isometry(Isometry3::translation(nalgebra::zero(),
                                                nalgebra::zero(),
                                                nalgebra::zero()))
            }
        }
        impl <N: RealField> Component for Isometry<N> {
            type
            Storage
            =
            VecStorage<Self>;
        }
    }
    pub mod graphics {
        use specs::{Component, VecStorage, FlaggedStorage};
        use vulkano::image::ImmutableImage;
        use vulkano::sampler::Sampler;
        use vulkano::format::Format;
        use vulkano::pipeline::GraphicsPipelineAbstract;
        use std::sync::Arc;
        use vulkano::descriptor::{PipelineLayoutAbstract, DescriptorSet};
        use std::marker::PhantomData;
        use vulkano::buffer::{BufferAccess, TypedBufferAccess};
        use vulkano::descriptor::descriptor_set;
        use frozengame::model::Vertex;
        use std::collections::{HashSet, HashMap};
        pub struct MeshBuffer<VD,
                              ID>(pub Vec<Arc<BufferAccess + Send + Sync +
                                              'static>>,
                                  pub Arc<TypedBufferAccess<Content = [ID]> +
                                          Send + Sync + 'static>,
                                  PhantomData<VD>);
        impl <VD, ID> MeshBuffer<VD, ID> {
            pub fn from(v: Vec<Arc<BufferAccess + Send + Sync + 'static>>,
                        i:
                            Arc<TypedBufferAccess<Content = [ID]> + Send +
                                Sync + 'static>) -> Self {
                MeshBuffer(v, i, PhantomData::default())
            }
        }
        impl <VD, ID> Component for MeshBuffer<VD, ID> where
         MeshBuffer<VD, ID>: Send + Sync + 'static {
            type
            Storage
            =
            VecStorage<Self>;
        }
        pub struct Mesh<VD, IT>(pub frozengame::model::Mesh<VD, IT>);
        impl Component for Mesh<Vertex, u32> {
            type
            Storage
            =
            VecStorage<Self>;
        }
        pub struct Texture(Sampler, ImmutableImage<Format>);
        impl Component for Texture {
            type
            Storage
            =
            FlaggedStorage<Self, VecStorage<Self>>;
        }
        pub struct GraphicsPipeline(pub Arc<GraphicsPipelineAbstract + Send +
                                            Sync + 'static>);
        impl Component for GraphicsPipeline {
            type
            Storage
            =
            VecStorage<Self>;
        }
        pub struct FixedSizeDescriptorSetsPool(pub descriptor_set::FixedSizeDescriptorSetsPool<Arc<PipelineLayoutAbstract +
                                                                                                   Send +
                                                                                                   Sync>>);
        impl Component for FixedSizeDescriptorSetsPool {
            type
            Storage
            =
            VecStorage<Self>;
        }
        pub struct DescriptorSetsCollection(pub Vec<Arc<DescriptorSet + Send +
                                                        Sync>>);
        impl DescriptorSetsCollection {
            pub fn push_or_replace(&mut self, index: usize,
                                   descriptor_set:
                                       Arc<DescriptorSet + Send + Sync>) {
                if self.0.len() <= index {
                    self.0.insert(index, descriptor_set);
                } else { self.0[index] = descriptor_set; }
            }
        }
        impl Default for DescriptorSetsCollection {
            fn default() -> Self { DescriptorSetsCollection(Vec::default()) }
        }
        impl Component for DescriptorSetsCollection {
            type
            Storage
            =
            VecStorage<Self>;
        }
    }
    pub struct PressedKeys(pub Vec<VirtualKeyCode>);
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::default::Default for PressedKeys {
        #[inline]
        fn default() -> PressedKeys {
            PressedKeys(::std::default::Default::default())
        }
    }
    pub struct MovementSystem;
    impl Component for Camera<f32> {
        type
        Storage
        =
        VecStorage<Self>;
    }
}
mod teapot_vs {
    #[allow(unused_imports)]
    use std::sync::Arc;
    #[allow(unused_imports)]
    use std::vec::IntoIter as VecIntoIter;
    #[allow(unused_imports)]
    use vulkano::device::Device;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::DescriptorDesc;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::DescriptorDescTy;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::DescriptorBufferDesc;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::DescriptorImageDesc;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::DescriptorImageDescDimensions;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::DescriptorImageDescArray;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::ShaderStages;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor_set::DescriptorSet;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor_set::UnsafeDescriptorSet;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
    #[allow(unused_imports)]
    use vulkano::descriptor::pipeline_layout::PipelineLayout;
    #[allow(unused_imports)]
    use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
    #[allow(unused_imports)]
    use vulkano::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
    #[allow(unused_imports)]
    use vulkano::pipeline::shader::SpecializationConstants as SpecConstsTrait;
    #[allow(unused_imports)]
    use vulkano::pipeline::shader::SpecializationMapEntry;
    pub struct Shader {
        shader: ::std::sync::Arc<::vulkano::pipeline::shader::ShaderModule>,
    }
    impl Shader {
        #[doc = r" Loads the shader in Vulkan as a `ShaderModule`."]
        #[inline]
        #[allow(unsafe_code)]
        pub fn load(device: ::std::sync::Arc<::vulkano::device::Device>)
         -> Result<Shader, ::vulkano::OomError> {
            let words =
                [119734787u32, 65536u32, 851975u32, 89u32, 0u32, 131089u32,
                 1u32, 393227u32, 1u32, 1280527431u32, 1685353262u32,
                 808793134u32, 0u32, 196622u32, 0u32, 1u32, 851983u32, 0u32,
                 4u32, 1852399981u32, 0u32, 13u32, 34u32, 46u32, 48u32, 51u32,
                 64u32, 75u32, 78u32, 196611u32, 2u32, 450u32, 655364u32,
                 1197427783u32, 1279741775u32, 1885560645u32, 1953718128u32,
                 1600482425u32, 1701734764u32, 1919509599u32, 1769235301u32,
                 25974u32, 524292u32, 1197427783u32, 1279741775u32,
                 1852399429u32, 1685417059u32, 1768185701u32, 1952671090u32,
                 6649449u32, 262149u32, 4u32, 1852399981u32, 0u32, 393221u32,
                 11u32, 1348430951u32, 1700164197u32, 2019914866u32, 0u32,
                 393222u32, 11u32, 0u32, 1348430951u32, 1953067887u32,
                 7237481u32, 458758u32, 11u32, 1u32, 1348430951u32,
                 1953393007u32, 1702521171u32, 0u32, 458758u32, 11u32, 2u32,
                 1130327143u32, 1148217708u32, 1635021673u32, 6644590u32,
                 458758u32, 11u32, 3u32, 1130327143u32, 1147956341u32,
                 1635021673u32, 6644590u32, 196613u32, 13u32, 0u32, 458757u32,
                 18u32, 1718185557u32, 1114468975u32, 1701209717u32,
                 1784827762u32, 7627621u32, 327686u32, 18u32, 0u32,
                 1349345635u32, 29551u32, 327686u32, 18u32, 1u32,
                 1701080941u32, 108u32, 327686u32, 18u32, 2u32, 2003134838u32,
                 0u32, 327686u32, 18u32, 3u32, 1785688688u32, 0u32, 196613u32,
                 20u32, 7299701u32, 327685u32, 34u32, 1769172848u32,
                 1852795252u32, 0u32, 393221u32, 46u32, 1734439526u32,
                 1131963732u32, 1685221231u32, 0u32, 327685u32, 48u32,
                 1601725812u32, 1919905635u32, 29540u32, 327685u32, 51u32,
                 1819438967u32, 1936674916u32, 0u32, 262149u32, 64u32,
                 1836216174u32, 27745u32, 262149u32, 75u32, 1836216174u32,
                 7564385u32, 262149u32, 78u32, 1601003875u32, 7565168u32,
                 327685u32, 85u32, 1400399220u32, 1819307361u32, 29285u32,
                 458757u32, 86u32, 1735749490u32, 1936027240u32,
                 1835094899u32, 1919249520u32, 0u32, 393221u32, 87u32,
                 1936289125u32, 1702259059u32, 1886216531u32, 7497068u32,
                 393221u32, 88u32, 1836216174u32, 1632857185u32,
                 1701605485u32, 114u32, 327752u32, 11u32, 0u32, 11u32, 0u32,
                 327752u32, 11u32, 1u32, 11u32, 1u32, 327752u32, 11u32, 2u32,
                 11u32, 3u32, 327752u32, 11u32, 3u32, 11u32, 4u32, 196679u32,
                 11u32, 2u32, 327752u32, 18u32, 0u32, 35u32, 0u32, 262216u32,
                 18u32, 1u32, 5u32, 327752u32, 18u32, 1u32, 35u32, 16u32,
                 327752u32, 18u32, 1u32, 7u32, 16u32, 262216u32, 18u32, 2u32,
                 5u32, 327752u32, 18u32, 2u32, 35u32, 80u32, 327752u32, 18u32,
                 2u32, 7u32, 16u32, 262216u32, 18u32, 3u32, 5u32, 327752u32,
                 18u32, 3u32, 35u32, 144u32, 327752u32, 18u32, 3u32, 7u32,
                 16u32, 196679u32, 18u32, 3u32, 262215u32, 20u32, 34u32, 1u32,
                 262215u32, 20u32, 33u32, 0u32, 262215u32, 34u32, 30u32, 0u32,
                 262215u32, 46u32, 30u32, 0u32, 262215u32, 48u32, 30u32, 2u32,
                 262215u32, 51u32, 30u32, 1u32, 262215u32, 64u32, 30u32, 2u32,
                 262215u32, 75u32, 30u32, 1u32, 262215u32, 78u32, 30u32, 3u32,
                 262215u32, 85u32, 34u32, 0u32, 262215u32, 85u32, 33u32, 0u32,
                 262215u32, 86u32, 34u32, 0u32, 262215u32, 86u32, 33u32, 1u32,
                 262215u32, 87u32, 34u32, 0u32, 262215u32, 87u32, 33u32, 2u32,
                 262215u32, 88u32, 34u32, 0u32, 262215u32, 88u32, 33u32, 3u32,
                 131091u32, 2u32, 196641u32, 3u32, 2u32, 196630u32, 6u32,
                 32u32, 262167u32, 7u32, 6u32, 4u32, 262165u32, 8u32, 32u32,
                 0u32, 262187u32, 8u32, 9u32, 1u32, 262172u32, 10u32, 6u32,
                 9u32, 393246u32, 11u32, 7u32, 6u32, 10u32, 10u32, 262176u32,
                 12u32, 3u32, 11u32, 262203u32, 12u32, 13u32, 3u32, 262165u32,
                 14u32, 32u32, 1u32, 262187u32, 14u32, 15u32, 0u32, 262167u32,
                 16u32, 6u32, 3u32, 262168u32, 17u32, 7u32, 4u32, 393246u32,
                 18u32, 16u32, 17u32, 17u32, 17u32, 262176u32, 19u32, 2u32,
                 18u32, 262203u32, 19u32, 20u32, 2u32, 262187u32, 14u32,
                 21u32, 3u32, 262176u32, 22u32, 2u32, 17u32, 262187u32, 14u32,
                 25u32, 2u32, 262187u32, 14u32, 29u32, 1u32, 262176u32, 33u32,
                 1u32, 16u32, 262203u32, 33u32, 34u32, 1u32, 262187u32, 6u32,
                 36u32, 1065353216u32, 262176u32, 42u32, 3u32, 7u32,
                 262167u32, 44u32, 6u32, 2u32, 262176u32, 45u32, 3u32, 44u32,
                 262203u32, 45u32, 46u32, 3u32, 262176u32, 47u32, 1u32, 44u32,
                 262203u32, 47u32, 48u32, 1u32, 262176u32, 50u32, 3u32, 16u32,
                 262203u32, 50u32, 51u32, 3u32, 262203u32, 50u32, 64u32, 3u32,
                 262168u32, 67u32, 16u32, 3u32, 262203u32, 33u32, 75u32, 1u32,
                 262203u32, 50u32, 78u32, 3u32, 262176u32, 79u32, 2u32, 16u32,
                 589849u32, 82u32, 6u32, 1u32, 0u32, 0u32, 0u32, 1u32, 0u32,
                 196635u32, 83u32, 82u32, 262176u32, 84u32, 0u32, 83u32,
                 262203u32, 84u32, 85u32, 0u32, 262203u32, 84u32, 86u32, 0u32,
                 262203u32, 84u32, 87u32, 0u32, 262203u32, 84u32, 88u32, 0u32,
                 327734u32, 2u32, 4u32, 0u32, 3u32, 131320u32, 5u32,
                 327745u32, 22u32, 23u32, 20u32, 21u32, 262205u32, 17u32,
                 24u32, 23u32, 327745u32, 22u32, 26u32, 20u32, 25u32,
                 262205u32, 17u32, 27u32, 26u32, 327826u32, 17u32, 28u32,
                 24u32, 27u32, 327745u32, 22u32, 30u32, 20u32, 29u32,
                 262205u32, 17u32, 31u32, 30u32, 327826u32, 17u32, 32u32,
                 28u32, 31u32, 262205u32, 16u32, 35u32, 34u32, 327761u32,
                 6u32, 37u32, 35u32, 0u32, 327761u32, 6u32, 38u32, 35u32,
                 1u32, 327761u32, 6u32, 39u32, 35u32, 2u32, 458832u32, 7u32,
                 40u32, 37u32, 38u32, 39u32, 36u32, 327825u32, 7u32, 41u32,
                 32u32, 40u32, 327745u32, 42u32, 43u32, 13u32, 15u32,
                 196670u32, 43u32, 41u32, 262205u32, 44u32, 49u32, 48u32,
                 196670u32, 46u32, 49u32, 327745u32, 22u32, 52u32, 20u32,
                 29u32, 262205u32, 17u32, 53u32, 52u32, 262205u32, 16u32,
                 54u32, 34u32, 327761u32, 6u32, 55u32, 54u32, 0u32, 327761u32,
                 6u32, 56u32, 54u32, 1u32, 327761u32, 6u32, 57u32, 54u32,
                 2u32, 458832u32, 7u32, 58u32, 55u32, 56u32, 57u32, 36u32,
                 327825u32, 7u32, 59u32, 53u32, 58u32, 327761u32, 6u32, 60u32,
                 59u32, 0u32, 327761u32, 6u32, 61u32, 59u32, 1u32, 327761u32,
                 6u32, 62u32, 59u32, 2u32, 393296u32, 16u32, 63u32, 60u32,
                 61u32, 62u32, 196670u32, 51u32, 63u32, 327745u32, 22u32,
                 65u32, 20u32, 29u32, 262205u32, 17u32, 66u32, 65u32,
                 327761u32, 7u32, 68u32, 66u32, 0u32, 524367u32, 16u32, 69u32,
                 68u32, 68u32, 0u32, 1u32, 2u32, 327761u32, 7u32, 70u32,
                 66u32, 1u32, 524367u32, 16u32, 71u32, 70u32, 70u32, 0u32,
                 1u32, 2u32, 327761u32, 7u32, 72u32, 66u32, 2u32, 524367u32,
                 16u32, 73u32, 72u32, 72u32, 0u32, 1u32, 2u32, 393296u32,
                 67u32, 74u32, 69u32, 71u32, 73u32, 262205u32, 16u32, 76u32,
                 75u32, 327825u32, 16u32, 77u32, 74u32, 76u32, 196670u32,
                 64u32, 77u32, 327745u32, 79u32, 80u32, 20u32, 15u32,
                 262205u32, 16u32, 81u32, 80u32, 196670u32, 78u32, 81u32,
                 65789u32, 65592u32];
            unsafe {
                Ok(Shader{shader:
                              match ::vulkano::pipeline::shader::ShaderModule::from_words(device,
                                                                                          &words)
                                  {
                                  ::core::result::Result::Ok(val) => val,
                                  ::core::result::Result::Err(err) => {
                                      return ::core::result::Result::Err(::core::convert::From::from(err))
                                  }
                              },})
            }
        }
        #[doc = r" Returns the module that was created."]
        #[allow(dead_code)]
        #[inline]
        pub fn module(&self)
         -> &::std::sync::Arc<::vulkano::pipeline::shader::ShaderModule> {
            &self.shader
        }
        #[doc =
              r" Returns a logical struct describing the entry point named `{ep_name}`."]
        #[inline]
        #[allow(unsafe_code)]
        pub fn main_entry_point(&self)
         ->
             ::vulkano::pipeline::shader::GraphicsEntryPoint<(), MainInput,
                                                             MainOutput,
                                                             Layout> {
            unsafe {
                #[allow(dead_code)]
                static NAME: [u8; 5usize] = [109u8, 97u8, 105u8, 110u8, 0];
                self.shader.graphics_entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr()
                                                                                as
                                                                                *const _),
                                                 MainInput, MainOutput,
                                                 Layout(ShaderStages{vertex:
                                                                         true,
                                                                                 ..ShaderStages::none()}),
                                                 ::vulkano::pipeline::shader::GraphicsShaderType::Vertex)
            }
        }
    }
    #[structural_match]
    #[rustc_copy_clone_marker]
    pub struct MainInput;
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::fmt::Debug for MainInput {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            match *self {
                MainInput => {
                    let mut debug_trait_builder = f.debug_tuple("MainInput");
                    debug_trait_builder.finish()
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::marker::Copy for MainInput { }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::clone::Clone for MainInput {
        #[inline]
        fn clone(&self) -> MainInput { { *self } }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::cmp::PartialEq for MainInput {
        #[inline]
        fn eq(&self, other: &MainInput) -> bool {
            match *other { MainInput => match *self { MainInput => true, }, }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::cmp::Eq for MainInput {
        #[inline]
        #[doc(hidden)]
        fn assert_receiver_is_total_eq(&self) -> () { { } }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::hash::Hash for MainInput {
        fn hash<__H: ::std::hash::Hasher>(&self, state: &mut __H) -> () {
            match *self { MainInput => { } }
        }
    }
    #[allow(unsafe_code)]
    unsafe impl ::vulkano::pipeline::shader::ShaderInterfaceDef for MainInput
     {
        type
        Iter
        =
        MainInputIter;
        fn elements(&self) -> MainInputIter { MainInputIter{num: 0,} }
    }
    #[rustc_copy_clone_marker]
    pub struct MainInputIter {
        num: u16,
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::fmt::Debug for MainInputIter {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            match *self {
                MainInputIter { num: ref __self_0_0 } => {
                    let mut debug_trait_builder =
                        f.debug_struct("MainInputIter");
                    let _ = debug_trait_builder.field("num", &&(*__self_0_0));
                    debug_trait_builder.finish()
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::marker::Copy for MainInputIter { }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::clone::Clone for MainInputIter {
        #[inline]
        fn clone(&self) -> MainInputIter {
            { let _: ::std::clone::AssertParamIsClone<u16>; *self }
        }
    }
    impl Iterator for MainInputIter {
        type
        Item
        =
        ::vulkano::pipeline::shader::ShaderInterfaceDefEntry;
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if self.num == 0u16 {
                self.num += 1;
                return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                     0u32..1u32,
                                                                                 format:
                                                                                     ::vulkano::format::Format::R32G32B32Sfloat,
                                                                                 name:
                                                                                     Some(::std::borrow::Cow::Borrowed("position")),});
            }
            if self.num == 1u16 {
                self.num += 1;
                return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                     2u32..3u32,
                                                                                 format:
                                                                                     ::vulkano::format::Format::R32G32Sfloat,
                                                                                 name:
                                                                                     Some(::std::borrow::Cow::Borrowed("tex_coords")),});
            }
            if self.num == 2u16 {
                self.num += 1;
                return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                     1u32..2u32,
                                                                                 format:
                                                                                     ::vulkano::format::Format::R32G32B32Sfloat,
                                                                                 name:
                                                                                     Some(::std::borrow::Cow::Borrowed("normals")),});
            }
            None
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = 3usize - self.num as usize;
            (len, Some(len))
        }
    }
    impl ExactSizeIterator for MainInputIter { }
    #[structural_match]
    #[rustc_copy_clone_marker]
    pub struct MainOutput;
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::fmt::Debug for MainOutput {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            match *self {
                MainOutput => {
                    let mut debug_trait_builder = f.debug_tuple("MainOutput");
                    debug_trait_builder.finish()
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::marker::Copy for MainOutput { }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::clone::Clone for MainOutput {
        #[inline]
        fn clone(&self) -> MainOutput { { *self } }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::cmp::PartialEq for MainOutput {
        #[inline]
        fn eq(&self, other: &MainOutput) -> bool {
            match *other {
                MainOutput => match *self { MainOutput => true, },
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::cmp::Eq for MainOutput {
        #[inline]
        #[doc(hidden)]
        fn assert_receiver_is_total_eq(&self) -> () { { } }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::hash::Hash for MainOutput {
        fn hash<__H: ::std::hash::Hasher>(&self, state: &mut __H) -> () {
            match *self { MainOutput => { } }
        }
    }
    #[allow(unsafe_code)]
    unsafe impl ::vulkano::pipeline::shader::ShaderInterfaceDef for MainOutput
     {
        type
        Iter
        =
        MainOutputIter;
        fn elements(&self) -> MainOutputIter { MainOutputIter{num: 0,} }
    }
    #[rustc_copy_clone_marker]
    pub struct MainOutputIter {
        num: u16,
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::fmt::Debug for MainOutputIter {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            match *self {
                MainOutputIter { num: ref __self_0_0 } => {
                    let mut debug_trait_builder =
                        f.debug_struct("MainOutputIter");
                    let _ = debug_trait_builder.field("num", &&(*__self_0_0));
                    debug_trait_builder.finish()
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::marker::Copy for MainOutputIter { }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::clone::Clone for MainOutputIter {
        #[inline]
        fn clone(&self) -> MainOutputIter {
            { let _: ::std::clone::AssertParamIsClone<u16>; *self }
        }
    }
    impl Iterator for MainOutputIter {
        type
        Item
        =
        ::vulkano::pipeline::shader::ShaderInterfaceDefEntry;
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if self.num == 0u16 {
                self.num += 1;
                return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                     0u32..1u32,
                                                                                 format:
                                                                                     ::vulkano::format::Format::R32G32Sfloat,
                                                                                 name:
                                                                                     Some(::std::borrow::Cow::Borrowed("fragTexCoord")),});
            }
            if self.num == 1u16 {
                self.num += 1;
                return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                     1u32..2u32,
                                                                                 format:
                                                                                     ::vulkano::format::Format::R32G32B32Sfloat,
                                                                                 name:
                                                                                     Some(::std::borrow::Cow::Borrowed("worldPos")),});
            }
            if self.num == 2u16 {
                self.num += 1;
                return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                     2u32..3u32,
                                                                                 format:
                                                                                     ::vulkano::format::Format::R32G32B32Sfloat,
                                                                                 name:
                                                                                     Some(::std::borrow::Cow::Borrowed("normal")),});
            }
            if self.num == 3u16 {
                self.num += 1;
                return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                     3u32..4u32,
                                                                                 format:
                                                                                     ::vulkano::format::Format::R32G32B32Sfloat,
                                                                                 name:
                                                                                     Some(::std::borrow::Cow::Borrowed("cam_pos")),});
            }
            None
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = 4usize - self.num as usize;
            (len, Some(len))
        }
    }
    impl ExactSizeIterator for MainOutputIter { }
    pub mod ty {
        #[repr(C)]
        #[allow(non_snake_case)]
        #[rustc_copy_clone_marker]
        pub struct UniformBufferObject {
            pub camPos: [f32; 3usize],
            pub _dummy0: [u8; 4usize],
            pub model: [[f32; 4usize]; 4usize],
            pub view: [[f32; 4usize]; 4usize],
            pub proj: [[f32; 4usize]; 4usize],
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        #[allow(non_snake_case)]
        impl ::std::marker::Copy for UniformBufferObject { }
        impl Clone for UniformBufferObject {
            fn clone(&self) -> Self {
                UniformBufferObject{camPos: self.camPos,
                                    _dummy0: self._dummy0,
                                    model: self.model,
                                    view: self.view,
                                    proj: self.proj,}
            }
        }
    }
    pub struct Layout(pub ShaderStages);
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::fmt::Debug for Layout {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            match *self {
                Layout(ref __self_0_0) => {
                    let mut debug_trait_builder = f.debug_tuple("Layout");
                    let _ = debug_trait_builder.field(&&(*__self_0_0));
                    debug_trait_builder.finish()
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::clone::Clone for Layout {
        #[inline]
        fn clone(&self) -> Layout {
            match *self {
                Layout(ref __self_0_0) =>
                Layout(::std::clone::Clone::clone(&(*__self_0_0))),
            }
        }
    }
    #[allow(unsafe_code)]
    unsafe impl PipelineLayoutDesc for Layout {
        fn num_sets(&self) -> usize { 2usize }
        fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
            match set {
                0usize => Some(4usize),
                1usize => Some(1usize),
                _ => None,
            }
        }
        fn descriptor(&self, set: usize, binding: usize)
         -> Option<DescriptorDesc> {
            match (set, binding) {
                (1usize, 0usize) =>
                Some(DescriptorDesc{ty:
                                        DescriptorDescTy::Buffer(DescriptorBufferDesc{dynamic:
                                                                                          Some(false),
                                                                                      storage:
                                                                                          true,}),
                                    array_count: 1u32,
                                    stages: self.0.clone(),
                                    readonly: true,}),
                (0usize, 0usize) =>
                Some(DescriptorDesc{ty:
                                        DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc{sampled:
                                                                                                       true,
                                                                                                   dimensions:
                                                                                                       DescriptorImageDescDimensions::TwoDimensional,
                                                                                                   format:
                                                                                                       None,
                                                                                                   multisampled:
                                                                                                       false,
                                                                                                   array_layers:
                                                                                                       DescriptorImageDescArray::NonArrayed,}),
                                    array_count: 1u32,
                                    stages: self.0.clone(),
                                    readonly: true,}),
                (0usize, 1usize) =>
                Some(DescriptorDesc{ty:
                                        DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc{sampled:
                                                                                                       true,
                                                                                                   dimensions:
                                                                                                       DescriptorImageDescDimensions::TwoDimensional,
                                                                                                   format:
                                                                                                       None,
                                                                                                   multisampled:
                                                                                                       false,
                                                                                                   array_layers:
                                                                                                       DescriptorImageDescArray::NonArrayed,}),
                                    array_count: 1u32,
                                    stages: self.0.clone(),
                                    readonly: true,}),
                (0usize, 2usize) =>
                Some(DescriptorDesc{ty:
                                        DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc{sampled:
                                                                                                       true,
                                                                                                   dimensions:
                                                                                                       DescriptorImageDescDimensions::TwoDimensional,
                                                                                                   format:
                                                                                                       None,
                                                                                                   multisampled:
                                                                                                       false,
                                                                                                   array_layers:
                                                                                                       DescriptorImageDescArray::NonArrayed,}),
                                    array_count: 1u32,
                                    stages: self.0.clone(),
                                    readonly: true,}),
                (0usize, 3usize) =>
                Some(DescriptorDesc{ty:
                                        DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc{sampled:
                                                                                                       true,
                                                                                                   dimensions:
                                                                                                       DescriptorImageDescDimensions::TwoDimensional,
                                                                                                   format:
                                                                                                       None,
                                                                                                   multisampled:
                                                                                                       false,
                                                                                                   array_layers:
                                                                                                       DescriptorImageDescArray::NonArrayed,}),
                                    array_count: 1u32,
                                    stages: self.0.clone(),
                                    readonly: true,}),
                _ => None,
            }
        }
        fn num_push_constants_ranges(&self) -> usize { 0usize }
        fn push_constants_range(&self, num: usize)
         -> Option<PipelineLayoutDescPcRange> {
            if num != 0 || 0usize == 0 {
                None
            } else {
                Some(PipelineLayoutDescPcRange{offset: 0,
                                               size: 0usize,
                                               stages: ShaderStages::all(),})
            }
        }
    }
    #[allow(non_snake_case)]
    #[repr(C)]
    #[rustc_copy_clone_marker]
    pub struct SpecializationConstants {
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    #[allow(non_snake_case)]
    impl ::std::fmt::Debug for SpecializationConstants {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            match *self {
                SpecializationConstants {  } => {
                    let mut debug_trait_builder =
                        f.debug_struct("SpecializationConstants");
                    debug_trait_builder.finish()
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    #[allow(non_snake_case)]
    impl ::std::marker::Copy for SpecializationConstants { }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    #[allow(non_snake_case)]
    impl ::std::clone::Clone for SpecializationConstants {
        #[inline]
        fn clone(&self) -> SpecializationConstants { { *self } }
    }
    impl Default for SpecializationConstants {
        fn default() -> SpecializationConstants { SpecializationConstants{} }
    }
    unsafe impl SpecConstsTrait for SpecializationConstants {
        fn descriptors() -> &'static [SpecializationMapEntry] {
            static DESCRIPTORS: [SpecializationMapEntry; 0usize] = [];
            &DESCRIPTORS
        }
    }
}
mod teapot_fs {
    #[allow(unused_imports)]
    use std::sync::Arc;
    #[allow(unused_imports)]
    use std::vec::IntoIter as VecIntoIter;
    #[allow(unused_imports)]
    use vulkano::device::Device;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::DescriptorDesc;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::DescriptorDescTy;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::DescriptorBufferDesc;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::DescriptorImageDesc;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::DescriptorImageDescDimensions;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::DescriptorImageDescArray;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor::ShaderStages;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor_set::DescriptorSet;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor_set::UnsafeDescriptorSet;
    #[allow(unused_imports)]
    use vulkano::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
    #[allow(unused_imports)]
    use vulkano::descriptor::pipeline_layout::PipelineLayout;
    #[allow(unused_imports)]
    use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
    #[allow(unused_imports)]
    use vulkano::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
    #[allow(unused_imports)]
    use vulkano::pipeline::shader::SpecializationConstants as SpecConstsTrait;
    #[allow(unused_imports)]
    use vulkano::pipeline::shader::SpecializationMapEntry;
    pub struct Shader {
        shader: ::std::sync::Arc<::vulkano::pipeline::shader::ShaderModule>,
    }
    impl Shader {
        #[doc = r" Loads the shader in Vulkan as a `ShaderModule`."]
        #[inline]
        #[allow(unsafe_code)]
        pub fn load(device: ::std::sync::Arc<::vulkano::device::Device>)
         -> Result<Shader, ::vulkano::OomError> {
            let words =
                [119734787u32, 65536u32, 851975u32, 409u32, 0u32, 131089u32,
                 1u32, 393227u32, 1u32, 1280527431u32, 1685353262u32,
                 808793134u32, 0u32, 196622u32, 0u32, 1u32, 655375u32, 4u32,
                 4u32, 1852399981u32, 0u32, 44u32, 56u32, 70u32, 239u32,
                 402u32, 196624u32, 4u32, 7u32, 196611u32, 2u32, 450u32,
                 655364u32, 1197427783u32, 1279741775u32, 1885560645u32,
                 1953718128u32, 1600482425u32, 1701734764u32, 1919509599u32,
                 1769235301u32, 25974u32, 524292u32, 1197427783u32,
                 1279741775u32, 1852399429u32, 1685417059u32, 1768185701u32,
                 1952671090u32, 6649449u32, 262149u32, 4u32, 1852399981u32,
                 0u32, 458757u32, 9u32, 1316250983u32, 1634562671u32,
                 1869760108u32, 1885425005u32, 40u32, 589829u32, 17u32,
                 1953720644u32, 1969383794u32, 1852795252u32, 676874055u32,
                 993224310u32, 993224310u32, 3879270u32, 196613u32, 14u32,
                 78u32, 196613u32, 15u32, 72u32, 327685u32, 16u32,
                 1735749490u32, 1936027240u32, 115u32, 589829u32, 22u32,
                 1836016967u32, 2037544037u32, 1818780499u32, 1198220137u32,
                 1713920071u32, 828783409u32, 59u32, 262149u32, 20u32,
                 1953457230u32, 86u32, 327685u32, 21u32, 1735749490u32,
                 1936027240u32, 115u32, 655365u32, 29u32, 1836016967u32,
                 2037544037u32, 1953066323u32, 1719019624u32, 1719024435u32,
                 1719024435u32, 828783411u32, 59u32, 196613u32, 25u32, 78u32,
                 196613u32, 26u32, 86u32, 196613u32, 27u32, 76u32, 327685u32,
                 28u32, 1735749490u32, 1936027240u32, 115u32, 524293u32,
                 34u32, 1936028262u32, 1399612782u32, 1768712291u32,
                 1713924963u32, 1719024433u32, 15155u32, 327685u32, 32u32,
                 1416851299u32, 1635018088u32, 0u32, 196613u32, 33u32,
                 12358u32, 393221u32, 36u32, 1735287156u32, 1316253285u32,
                 1634562671u32, 108u32, 393221u32, 40u32, 1836216174u32,
                 1632857185u32, 1701605485u32, 114u32, 393221u32, 44u32,
                 1734439526u32, 1131963732u32, 1685221231u32, 0u32, 196613u32,
                 54u32, 12625u32, 327685u32, 56u32, 1819438967u32,
                 1936674916u32, 0u32, 196613u32, 59u32, 12881u32, 196613u32,
                 63u32, 3241075u32, 196613u32, 66u32, 3306611u32, 196613u32,
                 69u32, 78u32, 262149u32, 70u32, 1836216174u32, 27745u32,
                 196613u32, 73u32, 84u32, 196613u32, 86u32, 66u32, 196613u32,
                 94u32, 5128788u32, 196613u32, 118u32, 97u32, 196613u32,
                 122u32, 12897u32, 262149u32, 126u32, 1953457230u32, 72u32,
                 262149u32, 131u32, 1953457230u32, 12872u32, 196613u32,
                 135u32, 7171950u32, 262149u32, 137u32, 1869505892u32, 109u32,
                 196613u32, 153u32, 114u32, 196613u32, 156u32, 107u32,
                 196613u32, 162u32, 7171950u32, 262149u32, 164u32,
                 1869505892u32, 109u32, 262149u32, 176u32, 1953457230u32,
                 86u32, 262149u32, 181u32, 1953457230u32, 76u32, 262149u32,
                 186u32, 846751591u32, 0u32, 262149u32, 187u32, 1634886000u32,
                 109u32, 262149u32, 189u32, 1634886000u32, 109u32, 262149u32,
                 192u32, 829974375u32, 0u32, 262149u32, 193u32, 1634886000u32,
                 109u32, 262149u32, 195u32, 1634886000u32, 109u32, 262149u32,
                 215u32, 1700949089u32, 28516u32, 327685u32, 216u32,
                 1400399220u32, 1819307361u32, 29285u32, 327685u32, 224u32,
                 1635018093u32, 1667853420u32, 0u32, 458757u32, 225u32,
                 1735749490u32, 1936027240u32, 1835094899u32, 1919249520u32,
                 0u32, 327685u32, 231u32, 1735749490u32, 1936027240u32,
                 115u32, 196613u32, 236u32, 78u32, 196613u32, 238u32, 86u32,
                 262149u32, 239u32, 1601003875u32, 7565168u32, 196613u32,
                 244u32, 12358u32, 196613u32, 252u32, 28492u32, 196613u32,
                 256u32, 105u32, 196613u32, 267u32, 76u32, 327685u32, 268u32,
                 1751607628u32, 1784827764u32, 7627621u32, 393222u32, 268u32,
                 0u32, 1769172848u32, 1852795252u32, 0u32, 327686u32, 268u32,
                 1u32, 1869377379u32, 114u32, 262149u32, 271u32,
                 1751607660u32, 29556u32, 196613u32, 279u32, 72u32, 327685u32,
                 284u32, 1953720676u32, 1701015137u32, 0u32, 327685u32,
                 291u32, 1702130785u32, 1952544110u32, 7237481u32, 327685u32,
                 296u32, 1768186226u32, 1701015137u32, 0u32, 196613u32,
                 302u32, 4605006u32, 262149u32, 303u32, 1634886000u32, 109u32,
                 262149u32, 305u32, 1634886000u32, 109u32, 262149u32, 307u32,
                 1634886000u32, 109u32, 196613u32, 310u32, 71u32, 262149u32,
                 311u32, 1634886000u32, 109u32, 262149u32, 313u32,
                 1634886000u32, 109u32, 262149u32, 315u32, 1634886000u32,
                 109u32, 262149u32, 317u32, 1634886000u32, 109u32, 196613u32,
                 320u32, 70u32, 262149u32, 325u32, 1634886000u32, 109u32,
                 262149u32, 326u32, 1634886000u32, 109u32, 327685u32, 329u32,
                 1768779630u32, 1869898094u32, 114u32, 327685u32, 335u32,
                 1869505892u32, 1634625901u32, 7499636u32, 327685u32, 349u32,
                 1667592307u32, 1918987381u32, 0u32, 196613u32, 354u32,
                 21355u32, 196613u32, 356u32, 17515u32, 262149u32, 364u32,
                 1953457230u32, 76u32, 262149u32, 384u32, 1768058209u32,
                 7630437u32, 262149u32, 389u32, 1869377379u32, 114u32,
                 262149u32, 402u32, 1868783462u32, 7499628u32, 393221u32,
                 408u32, 1936289125u32, 1702259059u32, 1886216531u32,
                 7497068u32, 262215u32, 40u32, 34u32, 0u32, 262215u32, 40u32,
                 33u32, 3u32, 262215u32, 44u32, 30u32, 0u32, 262215u32, 56u32,
                 30u32, 1u32, 262215u32, 70u32, 30u32, 2u32, 262215u32,
                 216u32, 34u32, 0u32, 262215u32, 216u32, 33u32, 0u32,
                 262215u32, 225u32, 34u32, 0u32, 262215u32, 225u32, 33u32,
                 1u32, 262215u32, 239u32, 30u32, 3u32, 327752u32, 268u32,
                 0u32, 35u32, 0u32, 327752u32, 268u32, 1u32, 35u32, 16u32,
                 196679u32, 268u32, 2u32, 262215u32, 271u32, 34u32, 1u32,
                 262215u32, 271u32, 33u32, 1u32, 262215u32, 402u32, 30u32,
                 0u32, 262215u32, 408u32, 34u32, 0u32, 262215u32, 408u32,
                 33u32, 2u32, 131091u32, 2u32, 196641u32, 3u32, 2u32,
                 196630u32, 6u32, 32u32, 262167u32, 7u32, 6u32, 3u32,
                 196641u32, 8u32, 7u32, 262176u32, 11u32, 7u32, 7u32,
                 262176u32, 12u32, 7u32, 6u32, 393249u32, 13u32, 6u32, 11u32,
                 11u32, 12u32, 327713u32, 19u32, 6u32, 12u32, 12u32,
                 458785u32, 24u32, 6u32, 11u32, 11u32, 11u32, 12u32,
                 327713u32, 31u32, 7u32, 12u32, 11u32, 589849u32, 37u32, 6u32,
                 1u32, 0u32, 0u32, 0u32, 1u32, 0u32, 196635u32, 38u32, 37u32,
                 262176u32, 39u32, 0u32, 38u32, 262203u32, 39u32, 40u32, 0u32,
                 262167u32, 42u32, 6u32, 2u32, 262176u32, 43u32, 1u32, 42u32,
                 262203u32, 43u32, 44u32, 1u32, 262167u32, 46u32, 6u32, 4u32,
                 262187u32, 6u32, 49u32, 1073741824u32, 262187u32, 6u32,
                 51u32, 1065353216u32, 262176u32, 55u32, 1u32, 7u32,
                 262203u32, 55u32, 56u32, 1u32, 262176u32, 62u32, 7u32, 42u32,
                 262203u32, 55u32, 70u32, 1u32, 262165u32, 75u32, 32u32, 0u32,
                 262187u32, 75u32, 76u32, 1u32, 262168u32, 92u32, 7u32, 3u32,
                 262176u32, 93u32, 7u32, 92u32, 262187u32, 6u32, 98u32, 0u32,
                 262187u32, 6u32, 143u32, 1078530011u32, 262187u32, 6u32,
                 160u32, 1090519040u32, 262187u32, 6u32, 209u32,
                 1084227584u32, 262203u32, 39u32, 216u32, 0u32, 262187u32,
                 6u32, 221u32, 1074580685u32, 393260u32, 7u32, 222u32, 221u32,
                 221u32, 221u32, 262203u32, 39u32, 225u32, 0u32, 262187u32,
                 75u32, 229u32, 2u32, 262203u32, 55u32, 239u32, 1u32,
                 262187u32, 6u32, 245u32, 1025758986u32, 393260u32, 7u32,
                 246u32, 245u32, 245u32, 245u32, 393260u32, 7u32, 253u32,
                 98u32, 98u32, 98u32, 262165u32, 254u32, 32u32, 1u32,
                 262176u32, 255u32, 7u32, 254u32, 262187u32, 254u32, 257u32,
                 0u32, 262187u32, 254u32, 264u32, 1u32, 131092u32, 265u32,
                 262174u32, 268u32, 7u32, 7u32, 262172u32, 269u32, 268u32,
                 76u32, 262176u32, 270u32, 2u32, 269u32, 262203u32, 270u32,
                 271u32, 2u32, 262176u32, 273u32, 2u32, 7u32, 262187u32, 6u32,
                 336u32, 1082130432u32, 262187u32, 6u32, 347u32, 981668463u32,
                 393260u32, 7u32, 357u32, 51u32, 51u32, 51u32, 262187u32,
                 6u32, 385u32, 1022739087u32, 393260u32, 7u32, 386u32, 385u32,
                 385u32, 385u32, 262187u32, 6u32, 398u32, 1055439407u32,
                 393260u32, 7u32, 399u32, 398u32, 398u32, 398u32, 262176u32,
                 401u32, 3u32, 46u32, 262203u32, 401u32, 402u32, 3u32,
                 262203u32, 39u32, 408u32, 0u32, 327734u32, 2u32, 4u32, 0u32,
                 3u32, 131320u32, 5u32, 262203u32, 11u32, 215u32, 7u32,
                 262203u32, 12u32, 224u32, 7u32, 262203u32, 12u32, 231u32,
                 7u32, 262203u32, 11u32, 236u32, 7u32, 262203u32, 11u32,
                 238u32, 7u32, 262203u32, 11u32, 244u32, 7u32, 262203u32,
                 11u32, 252u32, 7u32, 262203u32, 255u32, 256u32, 7u32,
                 262203u32, 11u32, 267u32, 7u32, 262203u32, 11u32, 279u32,
                 7u32, 262203u32, 12u32, 284u32, 7u32, 262203u32, 12u32,
                 291u32, 7u32, 262203u32, 11u32, 296u32, 7u32, 262203u32,
                 12u32, 302u32, 7u32, 262203u32, 11u32, 303u32, 7u32,
                 262203u32, 11u32, 305u32, 7u32, 262203u32, 12u32, 307u32,
                 7u32, 262203u32, 12u32, 310u32, 7u32, 262203u32, 11u32,
                 311u32, 7u32, 262203u32, 11u32, 313u32, 7u32, 262203u32,
                 11u32, 315u32, 7u32, 262203u32, 12u32, 317u32, 7u32,
                 262203u32, 11u32, 320u32, 7u32, 262203u32, 12u32, 325u32,
                 7u32, 262203u32, 11u32, 326u32, 7u32, 262203u32, 11u32,
                 329u32, 7u32, 262203u32, 12u32, 335u32, 7u32, 262203u32,
                 11u32, 349u32, 7u32, 262203u32, 11u32, 354u32, 7u32,
                 262203u32, 11u32, 356u32, 7u32, 262203u32, 12u32, 364u32,
                 7u32, 262203u32, 11u32, 384u32, 7u32, 262203u32, 11u32,
                 389u32, 7u32, 262205u32, 38u32, 217u32, 216u32, 262205u32,
                 42u32, 218u32, 44u32, 327767u32, 46u32, 219u32, 217u32,
                 218u32, 524367u32, 7u32, 220u32, 219u32, 219u32, 0u32, 1u32,
                 2u32, 458764u32, 7u32, 223u32, 1u32, 26u32, 220u32, 222u32,
                 196670u32, 215u32, 223u32, 262205u32, 38u32, 226u32, 225u32,
                 262205u32, 42u32, 227u32, 44u32, 327767u32, 46u32, 228u32,
                 226u32, 227u32, 327761u32, 6u32, 230u32, 228u32, 2u32,
                 196670u32, 224u32, 230u32, 262205u32, 38u32, 232u32, 225u32,
                 262205u32, 42u32, 233u32, 44u32, 327767u32, 46u32, 234u32,
                 232u32, 233u32, 327761u32, 6u32, 235u32, 234u32, 1u32,
                 196670u32, 231u32, 235u32, 262201u32, 7u32, 237u32, 9u32,
                 196670u32, 236u32, 237u32, 262205u32, 7u32, 240u32, 239u32,
                 262205u32, 7u32, 241u32, 56u32, 327811u32, 7u32, 242u32,
                 240u32, 241u32, 393228u32, 7u32, 243u32, 1u32, 69u32, 242u32,
                 196670u32, 238u32, 243u32, 196670u32, 244u32, 246u32,
                 262205u32, 7u32, 247u32, 244u32, 262205u32, 7u32, 248u32,
                 215u32, 262205u32, 6u32, 249u32, 224u32, 393296u32, 7u32,
                 250u32, 249u32, 249u32, 249u32, 524300u32, 7u32, 251u32,
                 1u32, 46u32, 247u32, 248u32, 250u32, 196670u32, 244u32,
                 251u32, 196670u32, 252u32, 253u32, 196670u32, 256u32, 257u32,
                 131321u32, 258u32, 131320u32, 258u32, 262390u32, 260u32,
                 261u32, 0u32, 131321u32, 262u32, 131320u32, 262u32,
                 262205u32, 254u32, 263u32, 256u32, 327857u32, 265u32, 266u32,
                 263u32, 264u32, 262394u32, 266u32, 259u32, 260u32, 131320u32,
                 259u32, 262205u32, 254u32, 272u32, 256u32, 393281u32, 273u32,
                 274u32, 271u32, 272u32, 257u32, 262205u32, 7u32, 275u32,
                 274u32, 262205u32, 7u32, 276u32, 56u32, 327811u32, 7u32,
                 277u32, 275u32, 276u32, 393228u32, 7u32, 278u32, 1u32, 69u32,
                 277u32, 196670u32, 267u32, 278u32, 262205u32, 7u32, 280u32,
                 238u32, 262205u32, 7u32, 281u32, 267u32, 327809u32, 7u32,
                 282u32, 280u32, 281u32, 393228u32, 7u32, 283u32, 1u32, 69u32,
                 282u32, 196670u32, 279u32, 283u32, 262205u32, 254u32, 285u32,
                 256u32, 393281u32, 273u32, 286u32, 271u32, 285u32, 257u32,
                 262205u32, 7u32, 287u32, 286u32, 262205u32, 7u32, 288u32,
                 56u32, 327811u32, 7u32, 289u32, 287u32, 288u32, 393228u32,
                 6u32, 290u32, 1u32, 66u32, 289u32, 196670u32, 284u32, 290u32,
                 262205u32, 6u32, 292u32, 284u32, 262205u32, 6u32, 293u32,
                 284u32, 327813u32, 6u32, 294u32, 292u32, 293u32, 327816u32,
                 6u32, 295u32, 51u32, 294u32, 196670u32, 291u32, 295u32,
                 262205u32, 254u32, 297u32, 256u32, 393281u32, 273u32, 298u32,
                 271u32, 297u32, 264u32, 262205u32, 7u32, 299u32, 298u32,
                 262205u32, 6u32, 300u32, 291u32, 327822u32, 7u32, 301u32,
                 299u32, 300u32, 196670u32, 296u32, 301u32, 262205u32, 7u32,
                 304u32, 236u32, 196670u32, 303u32, 304u32, 262205u32, 7u32,
                 306u32, 279u32, 196670u32, 305u32, 306u32, 262205u32, 6u32,
                 308u32, 231u32, 196670u32, 307u32, 308u32, 458809u32, 6u32,
                 309u32, 17u32, 303u32, 305u32, 307u32, 196670u32, 302u32,
                 309u32, 262205u32, 7u32, 312u32, 236u32, 196670u32, 311u32,
                 312u32, 262205u32, 7u32, 314u32, 238u32, 196670u32, 313u32,
                 314u32, 262205u32, 7u32, 316u32, 267u32, 196670u32, 315u32,
                 316u32, 262205u32, 6u32, 318u32, 231u32, 196670u32, 317u32,
                 318u32, 524345u32, 6u32, 319u32, 29u32, 311u32, 313u32,
                 315u32, 317u32, 196670u32, 310u32, 319u32, 262205u32, 7u32,
                 321u32, 279u32, 262205u32, 7u32, 322u32, 238u32, 327828u32,
                 6u32, 323u32, 321u32, 322u32, 458764u32, 6u32, 324u32, 1u32,
                 40u32, 323u32, 98u32, 196670u32, 325u32, 324u32, 262205u32,
                 7u32, 327u32, 244u32, 196670u32, 326u32, 327u32, 393273u32,
                 7u32, 328u32, 34u32, 325u32, 326u32, 196670u32, 320u32,
                 328u32, 262205u32, 6u32, 330u32, 302u32, 262205u32, 6u32,
                 331u32, 310u32, 327813u32, 6u32, 332u32, 330u32, 331u32,
                 262205u32, 7u32, 333u32, 320u32, 327822u32, 7u32, 334u32,
                 333u32, 332u32, 196670u32, 329u32, 334u32, 262205u32, 7u32,
                 337u32, 236u32, 262205u32, 7u32, 338u32, 238u32, 327828u32,
                 6u32, 339u32, 337u32, 338u32, 458764u32, 6u32, 340u32, 1u32,
                 40u32, 339u32, 98u32, 327813u32, 6u32, 341u32, 336u32,
                 340u32, 262205u32, 7u32, 342u32, 236u32, 262205u32, 7u32,
                 343u32, 267u32, 327828u32, 6u32, 344u32, 342u32, 343u32,
                 458764u32, 6u32, 345u32, 1u32, 40u32, 344u32, 98u32,
                 327813u32, 6u32, 346u32, 341u32, 345u32, 327809u32, 6u32,
                 348u32, 346u32, 347u32, 196670u32, 335u32, 348u32, 262205u32,
                 7u32, 350u32, 329u32, 262205u32, 6u32, 351u32, 335u32,
                 393296u32, 7u32, 352u32, 351u32, 351u32, 351u32, 327816u32,
                 7u32, 353u32, 350u32, 352u32, 196670u32, 349u32, 353u32,
                 262205u32, 7u32, 355u32, 320u32, 196670u32, 354u32, 355u32,
                 262205u32, 7u32, 358u32, 354u32, 327811u32, 7u32, 359u32,
                 357u32, 358u32, 196670u32, 356u32, 359u32, 262205u32, 6u32,
                 360u32, 224u32, 327811u32, 6u32, 361u32, 51u32, 360u32,
                 262205u32, 7u32, 362u32, 356u32, 327822u32, 7u32, 363u32,
                 362u32, 361u32, 196670u32, 356u32, 363u32, 262205u32, 7u32,
                 365u32, 236u32, 262205u32, 7u32, 366u32, 267u32, 327828u32,
                 6u32, 367u32, 365u32, 366u32, 458764u32, 6u32, 368u32, 1u32,
                 40u32, 367u32, 98u32, 196670u32, 364u32, 368u32, 262205u32,
                 7u32, 369u32, 356u32, 262205u32, 7u32, 370u32, 215u32,
                 327813u32, 7u32, 371u32, 369u32, 370u32, 393296u32, 7u32,
                 372u32, 143u32, 143u32, 143u32, 327816u32, 7u32, 373u32,
                 371u32, 372u32, 262205u32, 7u32, 374u32, 349u32, 327809u32,
                 7u32, 375u32, 373u32, 374u32, 262205u32, 7u32, 376u32,
                 296u32, 327813u32, 7u32, 377u32, 375u32, 376u32, 262205u32,
                 6u32, 378u32, 364u32, 327822u32, 7u32, 379u32, 377u32,
                 378u32, 262205u32, 7u32, 380u32, 252u32, 327809u32, 7u32,
                 381u32, 380u32, 379u32, 196670u32, 252u32, 381u32, 131321u32,
                 261u32, 131320u32, 261u32, 262205u32, 254u32, 382u32, 256u32,
                 327808u32, 254u32, 383u32, 382u32, 264u32, 196670u32, 256u32,
                 383u32, 131321u32, 258u32, 131320u32, 260u32, 262205u32,
                 7u32, 387u32, 215u32, 327813u32, 7u32, 388u32, 386u32,
                 387u32, 196670u32, 384u32, 388u32, 262205u32, 7u32, 390u32,
                 384u32, 262205u32, 7u32, 391u32, 252u32, 327809u32, 7u32,
                 392u32, 390u32, 391u32, 196670u32, 389u32, 392u32, 262205u32,
                 7u32, 393u32, 389u32, 262205u32, 7u32, 394u32, 389u32,
                 327809u32, 7u32, 395u32, 394u32, 357u32, 327816u32, 7u32,
                 396u32, 393u32, 395u32, 196670u32, 389u32, 396u32, 262205u32,
                 7u32, 397u32, 389u32, 458764u32, 7u32, 400u32, 1u32, 26u32,
                 397u32, 399u32, 196670u32, 389u32, 400u32, 262205u32, 7u32,
                 403u32, 389u32, 327761u32, 6u32, 404u32, 403u32, 0u32,
                 327761u32, 6u32, 405u32, 403u32, 1u32, 327761u32, 6u32,
                 406u32, 403u32, 2u32, 458832u32, 46u32, 407u32, 404u32,
                 405u32, 406u32, 51u32, 196670u32, 402u32, 407u32, 65789u32,
                 65592u32, 327734u32, 7u32, 9u32, 0u32, 8u32, 131320u32,
                 10u32, 262203u32, 11u32, 36u32, 7u32, 262203u32, 11u32,
                 54u32, 7u32, 262203u32, 11u32, 59u32, 7u32, 262203u32, 62u32,
                 63u32, 7u32, 262203u32, 62u32, 66u32, 7u32, 262203u32, 11u32,
                 69u32, 7u32, 262203u32, 11u32, 73u32, 7u32, 262203u32, 11u32,
                 86u32, 7u32, 262203u32, 93u32, 94u32, 7u32, 262205u32, 38u32,
                 41u32, 40u32, 262205u32, 42u32, 45u32, 44u32, 327767u32,
                 46u32, 47u32, 41u32, 45u32, 524367u32, 7u32, 48u32, 47u32,
                 47u32, 0u32, 1u32, 2u32, 327822u32, 7u32, 50u32, 48u32,
                 49u32, 393296u32, 7u32, 52u32, 51u32, 51u32, 51u32,
                 327811u32, 7u32, 53u32, 50u32, 52u32, 196670u32, 36u32,
                 53u32, 262205u32, 7u32, 57u32, 56u32, 262351u32, 7u32, 58u32,
                 57u32, 196670u32, 54u32, 58u32, 262205u32, 7u32, 60u32,
                 56u32, 262352u32, 7u32, 61u32, 60u32, 196670u32, 59u32,
                 61u32, 262205u32, 42u32, 64u32, 44u32, 262351u32, 42u32,
                 65u32, 64u32, 196670u32, 63u32, 65u32, 262205u32, 42u32,
                 67u32, 44u32, 262352u32, 42u32, 68u32, 67u32, 196670u32,
                 66u32, 68u32, 262205u32, 7u32, 71u32, 70u32, 393228u32, 7u32,
                 72u32, 1u32, 69u32, 71u32, 196670u32, 69u32, 72u32,
                 262205u32, 7u32, 74u32, 54u32, 327745u32, 12u32, 77u32,
                 66u32, 76u32, 262205u32, 6u32, 78u32, 77u32, 327822u32, 7u32,
                 79u32, 74u32, 78u32, 262205u32, 7u32, 80u32, 59u32,
                 327745u32, 12u32, 81u32, 63u32, 76u32, 262205u32, 6u32,
                 82u32, 81u32, 327822u32, 7u32, 83u32, 80u32, 82u32,
                 327811u32, 7u32, 84u32, 79u32, 83u32, 393228u32, 7u32, 85u32,
                 1u32, 69u32, 84u32, 196670u32, 73u32, 85u32, 262205u32, 7u32,
                 87u32, 69u32, 262205u32, 7u32, 88u32, 73u32, 458764u32, 7u32,
                 89u32, 1u32, 68u32, 87u32, 88u32, 393228u32, 7u32, 90u32,
                 1u32, 69u32, 89u32, 262271u32, 7u32, 91u32, 90u32, 196670u32,
                 86u32, 91u32, 262205u32, 7u32, 95u32, 73u32, 262205u32, 7u32,
                 96u32, 86u32, 262205u32, 7u32, 97u32, 69u32, 327761u32, 6u32,
                 99u32, 95u32, 0u32, 327761u32, 6u32, 100u32, 95u32, 1u32,
                 327761u32, 6u32, 101u32, 95u32, 2u32, 327761u32, 6u32,
                 102u32, 96u32, 0u32, 327761u32, 6u32, 103u32, 96u32, 1u32,
                 327761u32, 6u32, 104u32, 96u32, 2u32, 327761u32, 6u32,
                 105u32, 97u32, 0u32, 327761u32, 6u32, 106u32, 97u32, 1u32,
                 327761u32, 6u32, 107u32, 97u32, 2u32, 393296u32, 7u32,
                 108u32, 99u32, 100u32, 101u32, 393296u32, 7u32, 109u32,
                 102u32, 103u32, 104u32, 393296u32, 7u32, 110u32, 105u32,
                 106u32, 107u32, 393296u32, 92u32, 111u32, 108u32, 109u32,
                 110u32, 196670u32, 94u32, 111u32, 262205u32, 92u32, 112u32,
                 94u32, 262205u32, 7u32, 113u32, 36u32, 327825u32, 7u32,
                 114u32, 112u32, 113u32, 393228u32, 7u32, 115u32, 1u32, 69u32,
                 114u32, 131326u32, 115u32, 65592u32, 327734u32, 6u32, 17u32,
                 0u32, 13u32, 196663u32, 11u32, 14u32, 196663u32, 11u32,
                 15u32, 196663u32, 12u32, 16u32, 131320u32, 18u32, 262203u32,
                 12u32, 118u32, 7u32, 262203u32, 12u32, 122u32, 7u32,
                 262203u32, 12u32, 126u32, 7u32, 262203u32, 12u32, 131u32,
                 7u32, 262203u32, 12u32, 135u32, 7u32, 262203u32, 12u32,
                 137u32, 7u32, 262205u32, 6u32, 119u32, 16u32, 262205u32,
                 6u32, 120u32, 16u32, 327813u32, 6u32, 121u32, 119u32, 120u32,
                 196670u32, 118u32, 121u32, 262205u32, 6u32, 123u32, 118u32,
                 262205u32, 6u32, 124u32, 118u32, 327813u32, 6u32, 125u32,
                 123u32, 124u32, 196670u32, 122u32, 125u32, 262205u32, 7u32,
                 127u32, 14u32, 262205u32, 7u32, 128u32, 15u32, 327828u32,
                 6u32, 129u32, 127u32, 128u32, 458764u32, 6u32, 130u32, 1u32,
                 40u32, 129u32, 98u32, 196670u32, 126u32, 130u32, 262205u32,
                 6u32, 132u32, 126u32, 262205u32, 6u32, 133u32, 126u32,
                 327813u32, 6u32, 134u32, 132u32, 133u32, 196670u32, 131u32,
                 134u32, 262205u32, 6u32, 136u32, 122u32, 196670u32, 135u32,
                 136u32, 262205u32, 6u32, 138u32, 131u32, 262205u32, 6u32,
                 139u32, 122u32, 327811u32, 6u32, 140u32, 139u32, 51u32,
                 327813u32, 6u32, 141u32, 138u32, 140u32, 327809u32, 6u32,
                 142u32, 141u32, 51u32, 196670u32, 137u32, 142u32, 262205u32,
                 6u32, 144u32, 137u32, 327813u32, 6u32, 145u32, 143u32,
                 144u32, 262205u32, 6u32, 146u32, 137u32, 327813u32, 6u32,
                 147u32, 145u32, 146u32, 196670u32, 137u32, 147u32, 262205u32,
                 6u32, 148u32, 135u32, 262205u32, 6u32, 149u32, 137u32,
                 327816u32, 6u32, 150u32, 148u32, 149u32, 131326u32, 150u32,
                 65592u32, 327734u32, 6u32, 22u32, 0u32, 19u32, 196663u32,
                 12u32, 20u32, 196663u32, 12u32, 21u32, 131320u32, 23u32,
                 262203u32, 12u32, 153u32, 7u32, 262203u32, 12u32, 156u32,
                 7u32, 262203u32, 12u32, 162u32, 7u32, 262203u32, 12u32,
                 164u32, 7u32, 262205u32, 6u32, 154u32, 21u32, 327809u32,
                 6u32, 155u32, 154u32, 51u32, 196670u32, 153u32, 155u32,
                 262205u32, 6u32, 157u32, 153u32, 262205u32, 6u32, 158u32,
                 153u32, 327813u32, 6u32, 159u32, 157u32, 158u32, 327816u32,
                 6u32, 161u32, 159u32, 160u32, 196670u32, 156u32, 161u32,
                 262205u32, 6u32, 163u32, 20u32, 196670u32, 162u32, 163u32,
                 262205u32, 6u32, 165u32, 20u32, 262205u32, 6u32, 166u32,
                 156u32, 327811u32, 6u32, 167u32, 51u32, 166u32, 327813u32,
                 6u32, 168u32, 165u32, 167u32, 262205u32, 6u32, 169u32,
                 156u32, 327809u32, 6u32, 170u32, 168u32, 169u32, 196670u32,
                 164u32, 170u32, 262205u32, 6u32, 171u32, 162u32, 262205u32,
                 6u32, 172u32, 164u32, 327816u32, 6u32, 173u32, 171u32,
                 172u32, 131326u32, 173u32, 65592u32, 327734u32, 6u32, 29u32,
                 0u32, 24u32, 196663u32, 11u32, 25u32, 196663u32, 11u32,
                 26u32, 196663u32, 11u32, 27u32, 196663u32, 12u32, 28u32,
                 131320u32, 30u32, 262203u32, 12u32, 176u32, 7u32, 262203u32,
                 12u32, 181u32, 7u32, 262203u32, 12u32, 186u32, 7u32,
                 262203u32, 12u32, 187u32, 7u32, 262203u32, 12u32, 189u32,
                 7u32, 262203u32, 12u32, 192u32, 7u32, 262203u32, 12u32,
                 193u32, 7u32, 262203u32, 12u32, 195u32, 7u32, 262205u32,
                 7u32, 177u32, 25u32, 262205u32, 7u32, 178u32, 26u32,
                 327828u32, 6u32, 179u32, 177u32, 178u32, 458764u32, 6u32,
                 180u32, 1u32, 40u32, 179u32, 98u32, 196670u32, 176u32,
                 180u32, 262205u32, 7u32, 182u32, 25u32, 262205u32, 7u32,
                 183u32, 27u32, 327828u32, 6u32, 184u32, 182u32, 183u32,
                 458764u32, 6u32, 185u32, 1u32, 40u32, 184u32, 98u32,
                 196670u32, 181u32, 185u32, 262205u32, 6u32, 188u32, 176u32,
                 196670u32, 187u32, 188u32, 262205u32, 6u32, 190u32, 28u32,
                 196670u32, 189u32, 190u32, 393273u32, 6u32, 191u32, 22u32,
                 187u32, 189u32, 196670u32, 186u32, 191u32, 262205u32, 6u32,
                 194u32, 181u32, 196670u32, 193u32, 194u32, 262205u32, 6u32,
                 196u32, 28u32, 196670u32, 195u32, 196u32, 393273u32, 6u32,
                 197u32, 22u32, 193u32, 195u32, 196670u32, 192u32, 197u32,
                 262205u32, 6u32, 198u32, 192u32, 262205u32, 6u32, 199u32,
                 186u32, 327813u32, 6u32, 200u32, 198u32, 199u32, 131326u32,
                 200u32, 65592u32, 327734u32, 7u32, 34u32, 0u32, 31u32,
                 196663u32, 12u32, 32u32, 196663u32, 11u32, 33u32, 131320u32,
                 35u32, 262205u32, 7u32, 203u32, 33u32, 262205u32, 7u32,
                 204u32, 33u32, 393296u32, 7u32, 205u32, 51u32, 51u32, 51u32,
                 327811u32, 7u32, 206u32, 205u32, 204u32, 262205u32, 6u32,
                 207u32, 32u32, 327811u32, 6u32, 208u32, 51u32, 207u32,
                 458764u32, 6u32, 210u32, 1u32, 26u32, 208u32, 209u32,
                 327822u32, 7u32, 211u32, 206u32, 210u32, 327809u32, 7u32,
                 212u32, 203u32, 211u32, 131326u32, 212u32, 65592u32];
            unsafe {
                Ok(Shader{shader:
                              match ::vulkano::pipeline::shader::ShaderModule::from_words(device,
                                                                                          &words)
                                  {
                                  ::core::result::Result::Ok(val) => val,
                                  ::core::result::Result::Err(err) => {
                                      return ::core::result::Result::Err(::core::convert::From::from(err))
                                  }
                              },})
            }
        }
        #[doc = r" Returns the module that was created."]
        #[allow(dead_code)]
        #[inline]
        pub fn module(&self)
         -> &::std::sync::Arc<::vulkano::pipeline::shader::ShaderModule> {
            &self.shader
        }
        #[doc =
              r" Returns a logical struct describing the entry point named `{ep_name}`."]
        #[inline]
        #[allow(unsafe_code)]
        pub fn main_entry_point(&self)
         ->
             ::vulkano::pipeline::shader::GraphicsEntryPoint<(), MainInput,
                                                             MainOutput,
                                                             Layout> {
            unsafe {
                #[allow(dead_code)]
                static NAME: [u8; 5usize] = [109u8, 97u8, 105u8, 110u8, 0];
                self.shader.graphics_entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr()
                                                                                as
                                                                                *const _),
                                                 MainInput, MainOutput,
                                                 Layout(ShaderStages{fragment:
                                                                         true,
                                                                                 ..ShaderStages::none()}),
                                                 ::vulkano::pipeline::shader::GraphicsShaderType::Fragment)
            }
        }
    }
    #[structural_match]
    #[rustc_copy_clone_marker]
    pub struct MainInput;
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::fmt::Debug for MainInput {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            match *self {
                MainInput => {
                    let mut debug_trait_builder = f.debug_tuple("MainInput");
                    debug_trait_builder.finish()
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::marker::Copy for MainInput { }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::clone::Clone for MainInput {
        #[inline]
        fn clone(&self) -> MainInput { { *self } }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::cmp::PartialEq for MainInput {
        #[inline]
        fn eq(&self, other: &MainInput) -> bool {
            match *other { MainInput => match *self { MainInput => true, }, }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::cmp::Eq for MainInput {
        #[inline]
        #[doc(hidden)]
        fn assert_receiver_is_total_eq(&self) -> () { { } }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::hash::Hash for MainInput {
        fn hash<__H: ::std::hash::Hasher>(&self, state: &mut __H) -> () {
            match *self { MainInput => { } }
        }
    }
    #[allow(unsafe_code)]
    unsafe impl ::vulkano::pipeline::shader::ShaderInterfaceDef for MainInput
     {
        type
        Iter
        =
        MainInputIter;
        fn elements(&self) -> MainInputIter { MainInputIter{num: 0,} }
    }
    #[rustc_copy_clone_marker]
    pub struct MainInputIter {
        num: u16,
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::fmt::Debug for MainInputIter {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            match *self {
                MainInputIter { num: ref __self_0_0 } => {
                    let mut debug_trait_builder =
                        f.debug_struct("MainInputIter");
                    let _ = debug_trait_builder.field("num", &&(*__self_0_0));
                    debug_trait_builder.finish()
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::marker::Copy for MainInputIter { }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::clone::Clone for MainInputIter {
        #[inline]
        fn clone(&self) -> MainInputIter {
            { let _: ::std::clone::AssertParamIsClone<u16>; *self }
        }
    }
    impl Iterator for MainInputIter {
        type
        Item
        =
        ::vulkano::pipeline::shader::ShaderInterfaceDefEntry;
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if self.num == 0u16 {
                self.num += 1;
                return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                     0u32..1u32,
                                                                                 format:
                                                                                     ::vulkano::format::Format::R32G32Sfloat,
                                                                                 name:
                                                                                     Some(::std::borrow::Cow::Borrowed("fragTexCoord")),});
            }
            if self.num == 1u16 {
                self.num += 1;
                return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                     1u32..2u32,
                                                                                 format:
                                                                                     ::vulkano::format::Format::R32G32B32Sfloat,
                                                                                 name:
                                                                                     Some(::std::borrow::Cow::Borrowed("worldPos")),});
            }
            if self.num == 2u16 {
                self.num += 1;
                return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                     2u32..3u32,
                                                                                 format:
                                                                                     ::vulkano::format::Format::R32G32B32Sfloat,
                                                                                 name:
                                                                                     Some(::std::borrow::Cow::Borrowed("normal")),});
            }
            if self.num == 3u16 {
                self.num += 1;
                return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                     3u32..4u32,
                                                                                 format:
                                                                                     ::vulkano::format::Format::R32G32B32Sfloat,
                                                                                 name:
                                                                                     Some(::std::borrow::Cow::Borrowed("cam_pos")),});
            }
            None
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = 4usize - self.num as usize;
            (len, Some(len))
        }
    }
    impl ExactSizeIterator for MainInputIter { }
    #[structural_match]
    #[rustc_copy_clone_marker]
    pub struct MainOutput;
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::fmt::Debug for MainOutput {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            match *self {
                MainOutput => {
                    let mut debug_trait_builder = f.debug_tuple("MainOutput");
                    debug_trait_builder.finish()
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::marker::Copy for MainOutput { }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::clone::Clone for MainOutput {
        #[inline]
        fn clone(&self) -> MainOutput { { *self } }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::cmp::PartialEq for MainOutput {
        #[inline]
        fn eq(&self, other: &MainOutput) -> bool {
            match *other {
                MainOutput => match *self { MainOutput => true, },
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::cmp::Eq for MainOutput {
        #[inline]
        #[doc(hidden)]
        fn assert_receiver_is_total_eq(&self) -> () { { } }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::hash::Hash for MainOutput {
        fn hash<__H: ::std::hash::Hasher>(&self, state: &mut __H) -> () {
            match *self { MainOutput => { } }
        }
    }
    #[allow(unsafe_code)]
    unsafe impl ::vulkano::pipeline::shader::ShaderInterfaceDef for MainOutput
     {
        type
        Iter
        =
        MainOutputIter;
        fn elements(&self) -> MainOutputIter { MainOutputIter{num: 0,} }
    }
    #[rustc_copy_clone_marker]
    pub struct MainOutputIter {
        num: u16,
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::fmt::Debug for MainOutputIter {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            match *self {
                MainOutputIter { num: ref __self_0_0 } => {
                    let mut debug_trait_builder =
                        f.debug_struct("MainOutputIter");
                    let _ = debug_trait_builder.field("num", &&(*__self_0_0));
                    debug_trait_builder.finish()
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::marker::Copy for MainOutputIter { }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::clone::Clone for MainOutputIter {
        #[inline]
        fn clone(&self) -> MainOutputIter {
            { let _: ::std::clone::AssertParamIsClone<u16>; *self }
        }
    }
    impl Iterator for MainOutputIter {
        type
        Item
        =
        ::vulkano::pipeline::shader::ShaderInterfaceDefEntry;
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if self.num == 0u16 {
                self.num += 1;
                return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                     0u32..1u32,
                                                                                 format:
                                                                                     ::vulkano::format::Format::R32G32B32A32Sfloat,
                                                                                 name:
                                                                                     Some(::std::borrow::Cow::Borrowed("f_color")),});
            }
            None
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = 1usize - self.num as usize;
            (len, Some(len))
        }
    }
    impl ExactSizeIterator for MainOutputIter { }
    pub mod ty {
        #[repr(C)]
        #[allow(non_snake_case)]
        #[rustc_copy_clone_marker]
        pub struct LightObject {
            pub position: [f32; 3usize],
            pub _dummy0: [u8; 4usize],
            pub color: [f32; 3usize],
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        #[allow(non_snake_case)]
        impl ::std::marker::Copy for LightObject { }
        impl Clone for LightObject {
            fn clone(&self) -> Self {
                LightObject{position: self.position,
                            _dummy0: self._dummy0,
                            color: self.color,}
            }
        }
    }
    pub struct Layout(pub ShaderStages);
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::fmt::Debug for Layout {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            match *self {
                Layout(ref __self_0_0) => {
                    let mut debug_trait_builder = f.debug_tuple("Layout");
                    let _ = debug_trait_builder.field(&&(*__self_0_0));
                    debug_trait_builder.finish()
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::std::clone::Clone for Layout {
        #[inline]
        fn clone(&self) -> Layout {
            match *self {
                Layout(ref __self_0_0) =>
                Layout(::std::clone::Clone::clone(&(*__self_0_0))),
            }
        }
    }
    #[allow(unsafe_code)]
    unsafe impl PipelineLayoutDesc for Layout {
        fn num_sets(&self) -> usize { 2usize }
        fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
            match set {
                0usize => Some(4usize),
                1usize => Some(2usize),
                _ => None,
            }
        }
        fn descriptor(&self, set: usize, binding: usize)
         -> Option<DescriptorDesc> {
            match (set, binding) {
                (0usize, 3usize) =>
                Some(DescriptorDesc{ty:
                                        DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc{sampled:
                                                                                                       true,
                                                                                                   dimensions:
                                                                                                       DescriptorImageDescDimensions::TwoDimensional,
                                                                                                   format:
                                                                                                       None,
                                                                                                   multisampled:
                                                                                                       false,
                                                                                                   array_layers:
                                                                                                       DescriptorImageDescArray::NonArrayed,}),
                                    array_count: 1u32,
                                    stages: self.0.clone(),
                                    readonly: true,}),
                (0usize, 0usize) =>
                Some(DescriptorDesc{ty:
                                        DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc{sampled:
                                                                                                       true,
                                                                                                   dimensions:
                                                                                                       DescriptorImageDescDimensions::TwoDimensional,
                                                                                                   format:
                                                                                                       None,
                                                                                                   multisampled:
                                                                                                       false,
                                                                                                   array_layers:
                                                                                                       DescriptorImageDescArray::NonArrayed,}),
                                    array_count: 1u32,
                                    stages: self.0.clone(),
                                    readonly: true,}),
                (0usize, 1usize) =>
                Some(DescriptorDesc{ty:
                                        DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc{sampled:
                                                                                                       true,
                                                                                                   dimensions:
                                                                                                       DescriptorImageDescDimensions::TwoDimensional,
                                                                                                   format:
                                                                                                       None,
                                                                                                   multisampled:
                                                                                                       false,
                                                                                                   array_layers:
                                                                                                       DescriptorImageDescArray::NonArrayed,}),
                                    array_count: 1u32,
                                    stages: self.0.clone(),
                                    readonly: true,}),
                (1usize, 1usize) =>
                Some(DescriptorDesc{ty:
                                        DescriptorDescTy::Buffer(DescriptorBufferDesc{dynamic:
                                                                                          Some(false),
                                                                                      storage:
                                                                                          false,}),
                                    array_count: 1u32,
                                    stages: self.0.clone(),
                                    readonly: true,}),
                (0usize, 2usize) =>
                Some(DescriptorDesc{ty:
                                        DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc{sampled:
                                                                                                       true,
                                                                                                   dimensions:
                                                                                                       DescriptorImageDescDimensions::TwoDimensional,
                                                                                                   format:
                                                                                                       None,
                                                                                                   multisampled:
                                                                                                       false,
                                                                                                   array_layers:
                                                                                                       DescriptorImageDescArray::NonArrayed,}),
                                    array_count: 1u32,
                                    stages: self.0.clone(),
                                    readonly: true,}),
                _ => None,
            }
        }
        fn num_push_constants_ranges(&self) -> usize { 0usize }
        fn push_constants_range(&self, num: usize)
         -> Option<PipelineLayoutDescPcRange> {
            if num != 0 || 0usize == 0 {
                None
            } else {
                Some(PipelineLayoutDescPcRange{offset: 0,
                                               size: 0usize,
                                               stages: ShaderStages::all(),})
            }
        }
    }
    #[allow(non_snake_case)]
    #[repr(C)]
    #[rustc_copy_clone_marker]
    pub struct SpecializationConstants {
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    #[allow(non_snake_case)]
    impl ::std::fmt::Debug for SpecializationConstants {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            match *self {
                SpecializationConstants {  } => {
                    let mut debug_trait_builder =
                        f.debug_struct("SpecializationConstants");
                    debug_trait_builder.finish()
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    #[allow(non_snake_case)]
    impl ::std::marker::Copy for SpecializationConstants { }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    #[allow(non_snake_case)]
    impl ::std::clone::Clone for SpecializationConstants {
        #[inline]
        fn clone(&self) -> SpecializationConstants { { *self } }
    }
    impl Default for SpecializationConstants {
        fn default() -> SpecializationConstants { SpecializationConstants{} }
    }
    unsafe impl SpecConstsTrait for SpecializationConstants {
        fn descriptors() -> &'static [SpecializationMapEntry] {
            static DESCRIPTORS: [SpecializationMapEntry; 0usize] = [];
            &DESCRIPTORS
        }
    }
}
pub struct UniformBufferObject<N: RealField> {
    cam_pos: Vector3<N>,
    model: Matrix4<N>,
    view: Matrix4<N>,
    proj: Matrix4<N>,
}
#[automatically_derived]
#[allow(unused_qualifications)]
impl <N: ::std::clone::Clone + RealField> ::std::clone::Clone for
 UniformBufferObject<N> {
    #[inline]
    fn clone(&self) -> UniformBufferObject<N> {
        match *self {
            UniformBufferObject {
            cam_pos: ref __self_0_0,
            model: ref __self_0_1,
            view: ref __self_0_2,
            proj: ref __self_0_3 } =>
            UniformBufferObject{cam_pos:
                                    ::std::clone::Clone::clone(&(*__self_0_0)),
                                model:
                                    ::std::clone::Clone::clone(&(*__self_0_1)),
                                view:
                                    ::std::clone::Clone::clone(&(*__self_0_2)),
                                proj:
                                    ::std::clone::Clone::clone(&(*__self_0_3)),},
        }
    }
}
extern crate frozengame;
fn main() {
    let mut fuji =
        FujiBuilder::new().with_window().build().unwrap().with_graphics_queue().with_present_queue().with_swapchain().build().unwrap();
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
    world.add_resource(CpuBufferPool::new(device.clone(), BufferUsage::all())
                           as CpuBufferPool<UniformBufferObject<f32>>);
    world.add_resource(CpuBufferPool::new(device.clone(), BufferUsage::all())
                           as
                           CpuBufferPool<crate::teapot_fs::ty::LightObject>);
    let teapot_vs = teapot_vs::Shader::load(device.clone()).unwrap();
    let teapot_fs = teapot_fs::Shader::load(device.clone()).unwrap();
    mod cube_vs {
        #[allow(unused_imports)]
        use std::sync::Arc;
        #[allow(unused_imports)]
        use std::vec::IntoIter as VecIntoIter;
        #[allow(unused_imports)]
        use vulkano::device::Device;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorDescTy;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorBufferDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorImageDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorImageDescDimensions;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorImageDescArray;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::ShaderStages;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor_set::DescriptorSet;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor_set::UnsafeDescriptorSet;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
        #[allow(unused_imports)]
        use vulkano::descriptor::pipeline_layout::PipelineLayout;
        #[allow(unused_imports)]
        use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
        #[allow(unused_imports)]
        use vulkano::pipeline::shader::SpecializationConstants as
            SpecConstsTrait;
        #[allow(unused_imports)]
        use vulkano::pipeline::shader::SpecializationMapEntry;
        pub struct Shader {
            shader: ::std::sync::Arc<::vulkano::pipeline::shader::ShaderModule>,
        }
        impl Shader {
            #[doc = r" Loads the shader in Vulkan as a `ShaderModule`."]
            #[inline]
            #[allow(unsafe_code)]
            pub fn load(device: ::std::sync::Arc<::vulkano::device::Device>)
             -> Result<Shader, ::vulkano::OomError> {
                let words =
                    [119734787u32, 65536u32, 851975u32, 44u32, 0u32,
                     131089u32, 1u32, 393227u32, 1u32, 1280527431u32,
                     1685353262u32, 808793134u32, 0u32, 196622u32, 0u32, 1u32,
                     524303u32, 0u32, 4u32, 1852399981u32, 0u32, 13u32, 33u32,
                     43u32, 196611u32, 2u32, 450u32, 655364u32, 1197427783u32,
                     1279741775u32, 1885560645u32, 1953718128u32,
                     1600482425u32, 1701734764u32, 1919509599u32,
                     1769235301u32, 25974u32, 524292u32, 1197427783u32,
                     1279741775u32, 1852399429u32, 1685417059u32,
                     1768185701u32, 1952671090u32, 6649449u32, 262149u32,
                     4u32, 1852399981u32, 0u32, 393221u32, 11u32,
                     1348430951u32, 1700164197u32, 2019914866u32, 0u32,
                     393222u32, 11u32, 0u32, 1348430951u32, 1953067887u32,
                     7237481u32, 458758u32, 11u32, 1u32, 1348430951u32,
                     1953393007u32, 1702521171u32, 0u32, 458758u32, 11u32,
                     2u32, 1130327143u32, 1148217708u32, 1635021673u32,
                     6644590u32, 458758u32, 11u32, 3u32, 1130327143u32,
                     1147956341u32, 1635021673u32, 6644590u32, 196613u32,
                     13u32, 0u32, 458757u32, 17u32, 1718185557u32,
                     1114468975u32, 1701209717u32, 1784827762u32, 7627621u32,
                     327686u32, 17u32, 0u32, 1701080941u32, 108u32, 327686u32,
                     17u32, 1u32, 2003134838u32, 0u32, 327686u32, 17u32, 2u32,
                     1785688688u32, 0u32, 196613u32, 19u32, 7299701u32,
                     327685u32, 33u32, 1769172848u32, 1852795252u32, 0u32,
                     262149u32, 43u32, 1836216174u32, 7564385u32, 327752u32,
                     11u32, 0u32, 11u32, 0u32, 327752u32, 11u32, 1u32, 11u32,
                     1u32, 327752u32, 11u32, 2u32, 11u32, 3u32, 327752u32,
                     11u32, 3u32, 11u32, 4u32, 196679u32, 11u32, 2u32,
                     262216u32, 17u32, 0u32, 5u32, 327752u32, 17u32, 0u32,
                     35u32, 0u32, 327752u32, 17u32, 0u32, 7u32, 16u32,
                     262216u32, 17u32, 1u32, 5u32, 327752u32, 17u32, 1u32,
                     35u32, 64u32, 327752u32, 17u32, 1u32, 7u32, 16u32,
                     262216u32, 17u32, 2u32, 5u32, 327752u32, 17u32, 2u32,
                     35u32, 128u32, 327752u32, 17u32, 2u32, 7u32, 16u32,
                     196679u32, 17u32, 3u32, 262215u32, 19u32, 34u32, 0u32,
                     262215u32, 19u32, 33u32, 0u32, 262215u32, 33u32, 30u32,
                     0u32, 262215u32, 43u32, 30u32, 1u32, 131091u32, 2u32,
                     196641u32, 3u32, 2u32, 196630u32, 6u32, 32u32, 262167u32,
                     7u32, 6u32, 4u32, 262165u32, 8u32, 32u32, 0u32,
                     262187u32, 8u32, 9u32, 1u32, 262172u32, 10u32, 6u32,
                     9u32, 393246u32, 11u32, 7u32, 6u32, 10u32, 10u32,
                     262176u32, 12u32, 3u32, 11u32, 262203u32, 12u32, 13u32,
                     3u32, 262165u32, 14u32, 32u32, 1u32, 262187u32, 14u32,
                     15u32, 0u32, 262168u32, 16u32, 7u32, 4u32, 327710u32,
                     17u32, 16u32, 16u32, 16u32, 262176u32, 18u32, 2u32,
                     17u32, 262203u32, 18u32, 19u32, 2u32, 262187u32, 14u32,
                     20u32, 2u32, 262176u32, 21u32, 2u32, 16u32, 262187u32,
                     14u32, 24u32, 1u32, 262167u32, 31u32, 6u32, 3u32,
                     262176u32, 32u32, 1u32, 31u32, 262203u32, 32u32, 33u32,
                     1u32, 262187u32, 6u32, 35u32, 1065353216u32, 262176u32,
                     41u32, 3u32, 7u32, 262203u32, 32u32, 43u32, 1u32,
                     327734u32, 2u32, 4u32, 0u32, 3u32, 131320u32, 5u32,
                     327745u32, 21u32, 22u32, 19u32, 20u32, 262205u32, 16u32,
                     23u32, 22u32, 327745u32, 21u32, 25u32, 19u32, 24u32,
                     262205u32, 16u32, 26u32, 25u32, 327826u32, 16u32, 27u32,
                     23u32, 26u32, 327745u32, 21u32, 28u32, 19u32, 15u32,
                     262205u32, 16u32, 29u32, 28u32, 327826u32, 16u32, 30u32,
                     27u32, 29u32, 262205u32, 31u32, 34u32, 33u32, 327761u32,
                     6u32, 36u32, 34u32, 0u32, 327761u32, 6u32, 37u32, 34u32,
                     1u32, 327761u32, 6u32, 38u32, 34u32, 2u32, 458832u32,
                     7u32, 39u32, 36u32, 37u32, 38u32, 35u32, 327825u32, 7u32,
                     40u32, 30u32, 39u32, 327745u32, 41u32, 42u32, 13u32,
                     15u32, 196670u32, 42u32, 40u32, 65789u32, 65592u32];
                unsafe {
                    Ok(Shader{shader:
                                  match ::vulkano::pipeline::shader::ShaderModule::from_words(device,
                                                                                              &words)
                                      {
                                      ::core::result::Result::Ok(val) => val,
                                      ::core::result::Result::Err(err) => {
                                          return ::core::result::Result::Err(::core::convert::From::from(err))
                                      }
                                  },})
                }
            }
            #[doc = r" Returns the module that was created."]
            #[allow(dead_code)]
            #[inline]
            pub fn module(&self)
             -> &::std::sync::Arc<::vulkano::pipeline::shader::ShaderModule> {
                &self.shader
            }
            #[doc =
                  r" Returns a logical struct describing the entry point named `{ep_name}`."]
            #[inline]
            #[allow(unsafe_code)]
            pub fn main_entry_point(&self)
             ->
                 ::vulkano::pipeline::shader::GraphicsEntryPoint<(),
                                                                 MainInput,
                                                                 MainOutput,
                                                                 Layout> {
                unsafe {
                    #[allow(dead_code)]
                    static NAME: [u8; 5usize] =
                        [109u8, 97u8, 105u8, 110u8, 0];
                    self.shader.graphics_entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr()
                                                                                    as
                                                                                    *const _),
                                                     MainInput, MainOutput,
                                                     Layout(ShaderStages{vertex:
                                                                             true,
                                                                                     ..ShaderStages::none()}),
                                                     ::vulkano::pipeline::shader::GraphicsShaderType::Vertex)
                }
            }
        }
        #[structural_match]
        #[rustc_copy_clone_marker]
        pub struct MainInput;
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::fmt::Debug for MainInput {
            fn fmt(&self, f: &mut ::std::fmt::Formatter)
             -> ::std::fmt::Result {
                match *self {
                    MainInput => {
                        let mut debug_trait_builder =
                            f.debug_tuple("MainInput");
                        debug_trait_builder.finish()
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::marker::Copy for MainInput { }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::clone::Clone for MainInput {
            #[inline]
            fn clone(&self) -> MainInput { { *self } }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::cmp::PartialEq for MainInput {
            #[inline]
            fn eq(&self, other: &MainInput) -> bool {
                match *other {
                    MainInput => match *self { MainInput => true, },
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::cmp::Eq for MainInput {
            #[inline]
            #[doc(hidden)]
            fn assert_receiver_is_total_eq(&self) -> () { { } }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::hash::Hash for MainInput {
            fn hash<__H: ::std::hash::Hasher>(&self, state: &mut __H) -> () {
                match *self { MainInput => { } }
            }
        }
        #[allow(unsafe_code)]
        unsafe impl ::vulkano::pipeline::shader::ShaderInterfaceDef for
         MainInput {
            type
            Iter
            =
            MainInputIter;
            fn elements(&self) -> MainInputIter { MainInputIter{num: 0,} }
        }
        #[rustc_copy_clone_marker]
        pub struct MainInputIter {
            num: u16,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::fmt::Debug for MainInputIter {
            fn fmt(&self, f: &mut ::std::fmt::Formatter)
             -> ::std::fmt::Result {
                match *self {
                    MainInputIter { num: ref __self_0_0 } => {
                        let mut debug_trait_builder =
                            f.debug_struct("MainInputIter");
                        let _ =
                            debug_trait_builder.field("num", &&(*__self_0_0));
                        debug_trait_builder.finish()
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::marker::Copy for MainInputIter { }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::clone::Clone for MainInputIter {
            #[inline]
            fn clone(&self) -> MainInputIter {
                { let _: ::std::clone::AssertParamIsClone<u16>; *self }
            }
        }
        impl Iterator for MainInputIter {
            type
            Item
            =
            ::vulkano::pipeline::shader::ShaderInterfaceDefEntry;
            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                if self.num == 0u16 {
                    self.num += 1;
                    return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                         0u32..1u32,
                                                                                     format:
                                                                                         ::vulkano::format::Format::R32G32B32Sfloat,
                                                                                     name:
                                                                                         Some(::std::borrow::Cow::Borrowed("position")),});
                }
                if self.num == 1u16 {
                    self.num += 1;
                    return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                         1u32..2u32,
                                                                                     format:
                                                                                         ::vulkano::format::Format::R32G32B32Sfloat,
                                                                                     name:
                                                                                         Some(::std::borrow::Cow::Borrowed("normals")),});
                }
                None
            }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = 2usize - self.num as usize;
                (len, Some(len))
            }
        }
        impl ExactSizeIterator for MainInputIter { }
        #[structural_match]
        #[rustc_copy_clone_marker]
        pub struct MainOutput;
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::fmt::Debug for MainOutput {
            fn fmt(&self, f: &mut ::std::fmt::Formatter)
             -> ::std::fmt::Result {
                match *self {
                    MainOutput => {
                        let mut debug_trait_builder =
                            f.debug_tuple("MainOutput");
                        debug_trait_builder.finish()
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::marker::Copy for MainOutput { }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::clone::Clone for MainOutput {
            #[inline]
            fn clone(&self) -> MainOutput { { *self } }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::cmp::PartialEq for MainOutput {
            #[inline]
            fn eq(&self, other: &MainOutput) -> bool {
                match *other {
                    MainOutput => match *self { MainOutput => true, },
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::cmp::Eq for MainOutput {
            #[inline]
            #[doc(hidden)]
            fn assert_receiver_is_total_eq(&self) -> () { { } }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::hash::Hash for MainOutput {
            fn hash<__H: ::std::hash::Hasher>(&self, state: &mut __H) -> () {
                match *self { MainOutput => { } }
            }
        }
        #[allow(unsafe_code)]
        unsafe impl ::vulkano::pipeline::shader::ShaderInterfaceDef for
         MainOutput {
            type
            Iter
            =
            MainOutputIter;
            fn elements(&self) -> MainOutputIter { MainOutputIter{num: 0,} }
        }
        #[rustc_copy_clone_marker]
        pub struct MainOutputIter {
            num: u16,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::fmt::Debug for MainOutputIter {
            fn fmt(&self, f: &mut ::std::fmt::Formatter)
             -> ::std::fmt::Result {
                match *self {
                    MainOutputIter { num: ref __self_0_0 } => {
                        let mut debug_trait_builder =
                            f.debug_struct("MainOutputIter");
                        let _ =
                            debug_trait_builder.field("num", &&(*__self_0_0));
                        debug_trait_builder.finish()
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::marker::Copy for MainOutputIter { }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::clone::Clone for MainOutputIter {
            #[inline]
            fn clone(&self) -> MainOutputIter {
                { let _: ::std::clone::AssertParamIsClone<u16>; *self }
            }
        }
        impl Iterator for MainOutputIter {
            type
            Item
            =
            ::vulkano::pipeline::shader::ShaderInterfaceDefEntry;
            #[inline]
            fn next(&mut self) -> Option<Self::Item> { None }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = 0usize - self.num as usize;
                (len, Some(len))
            }
        }
        impl ExactSizeIterator for MainOutputIter { }
        pub mod ty {
            #[repr(C)]
            #[allow(non_snake_case)]
            #[rustc_copy_clone_marker]
            pub struct UniformBufferObject {
                pub model: [[f32; 4usize]; 4usize],
                pub view: [[f32; 4usize]; 4usize],
                pub proj: [[f32; 4usize]; 4usize],
            }
            #[automatically_derived]
            #[allow(unused_qualifications)]
            #[allow(non_snake_case)]
            impl ::std::marker::Copy for UniformBufferObject { }
            impl Clone for UniformBufferObject {
                fn clone(&self) -> Self {
                    UniformBufferObject{model: self.model,
                                        view: self.view,
                                        proj: self.proj,}
                }
            }
        }
        pub struct Layout(pub ShaderStages);
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::fmt::Debug for Layout {
            fn fmt(&self, f: &mut ::std::fmt::Formatter)
             -> ::std::fmt::Result {
                match *self {
                    Layout(ref __self_0_0) => {
                        let mut debug_trait_builder = f.debug_tuple("Layout");
                        let _ = debug_trait_builder.field(&&(*__self_0_0));
                        debug_trait_builder.finish()
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::clone::Clone for Layout {
            #[inline]
            fn clone(&self) -> Layout {
                match *self {
                    Layout(ref __self_0_0) =>
                    Layout(::std::clone::Clone::clone(&(*__self_0_0))),
                }
            }
        }
        #[allow(unsafe_code)]
        unsafe impl PipelineLayoutDesc for Layout {
            fn num_sets(&self) -> usize { 1usize }
            fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
                match set { 0usize => Some(1usize), _ => None, }
            }
            fn descriptor(&self, set: usize, binding: usize)
             -> Option<DescriptorDesc> {
                match (set, binding) {
                    (0usize, 0usize) =>
                    Some(DescriptorDesc{ty:
                                            DescriptorDescTy::Buffer(DescriptorBufferDesc{dynamic:
                                                                                              Some(false),
                                                                                          storage:
                                                                                              true,}),
                                        array_count: 1u32,
                                        stages: self.0.clone(),
                                        readonly: true,}),
                    _ => None,
                }
            }
            fn num_push_constants_ranges(&self) -> usize { 0usize }
            fn push_constants_range(&self, num: usize)
             -> Option<PipelineLayoutDescPcRange> {
                if num != 0 || 0usize == 0 {
                    None
                } else {
                    Some(PipelineLayoutDescPcRange{offset: 0,
                                                   size: 0usize,
                                                   stages:
                                                       ShaderStages::all(),})
                }
            }
        }
        #[allow(non_snake_case)]
        #[repr(C)]
        #[rustc_copy_clone_marker]
        pub struct SpecializationConstants {
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        #[allow(non_snake_case)]
        impl ::std::fmt::Debug for SpecializationConstants {
            fn fmt(&self, f: &mut ::std::fmt::Formatter)
             -> ::std::fmt::Result {
                match *self {
                    SpecializationConstants {  } => {
                        let mut debug_trait_builder =
                            f.debug_struct("SpecializationConstants");
                        debug_trait_builder.finish()
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        #[allow(non_snake_case)]
        impl ::std::marker::Copy for SpecializationConstants { }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        #[allow(non_snake_case)]
        impl ::std::clone::Clone for SpecializationConstants {
            #[inline]
            fn clone(&self) -> SpecializationConstants { { *self } }
        }
        impl Default for SpecializationConstants {
            fn default() -> SpecializationConstants {
                SpecializationConstants{}
            }
        }
        unsafe impl SpecConstsTrait for SpecializationConstants {
            fn descriptors() -> &'static [SpecializationMapEntry] {
                static DESCRIPTORS: [SpecializationMapEntry; 0usize] = [];
                &DESCRIPTORS
            }
        }
    }
    mod cube_fs {
        #[allow(unused_imports)]
        use std::sync::Arc;
        #[allow(unused_imports)]
        use std::vec::IntoIter as VecIntoIter;
        #[allow(unused_imports)]
        use vulkano::device::Device;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorDescTy;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorBufferDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorImageDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorImageDescDimensions;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorImageDescArray;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::ShaderStages;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor_set::DescriptorSet;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor_set::UnsafeDescriptorSet;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
        #[allow(unused_imports)]
        use vulkano::descriptor::pipeline_layout::PipelineLayout;
        #[allow(unused_imports)]
        use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
        #[allow(unused_imports)]
        use vulkano::pipeline::shader::SpecializationConstants as
            SpecConstsTrait;
        #[allow(unused_imports)]
        use vulkano::pipeline::shader::SpecializationMapEntry;
        pub struct Shader {
            shader: ::std::sync::Arc<::vulkano::pipeline::shader::ShaderModule>,
        }
        impl Shader {
            #[doc = r" Loads the shader in Vulkan as a `ShaderModule`."]
            #[inline]
            #[allow(unsafe_code)]
            pub fn load(device: ::std::sync::Arc<::vulkano::device::Device>)
             -> Result<Shader, ::vulkano::OomError> {
                let words =
                    [119734787u32, 65536u32, 851975u32, 12u32, 0u32,
                     131089u32, 1u32, 393227u32, 1u32, 1280527431u32,
                     1685353262u32, 808793134u32, 0u32, 196622u32, 0u32, 1u32,
                     393231u32, 4u32, 4u32, 1852399981u32, 0u32, 9u32,
                     196624u32, 4u32, 7u32, 196611u32, 2u32, 450u32,
                     655364u32, 1197427783u32, 1279741775u32, 1885560645u32,
                     1953718128u32, 1600482425u32, 1701734764u32,
                     1919509599u32, 1769235301u32, 25974u32, 524292u32,
                     1197427783u32, 1279741775u32, 1852399429u32,
                     1685417059u32, 1768185701u32, 1952671090u32, 6649449u32,
                     262149u32, 4u32, 1852399981u32, 0u32, 262149u32, 9u32,
                     1868783462u32, 7499628u32, 262215u32, 9u32, 30u32, 0u32,
                     131091u32, 2u32, 196641u32, 3u32, 2u32, 196630u32, 6u32,
                     32u32, 262167u32, 7u32, 6u32, 4u32, 262176u32, 8u32,
                     3u32, 7u32, 262203u32, 8u32, 9u32, 3u32, 262187u32, 6u32,
                     10u32, 1065353216u32, 458796u32, 7u32, 11u32, 10u32,
                     10u32, 10u32, 10u32, 327734u32, 2u32, 4u32, 0u32, 3u32,
                     131320u32, 5u32, 196670u32, 9u32, 11u32, 65789u32,
                     65592u32];
                unsafe {
                    Ok(Shader{shader:
                                  match ::vulkano::pipeline::shader::ShaderModule::from_words(device,
                                                                                              &words)
                                      {
                                      ::core::result::Result::Ok(val) => val,
                                      ::core::result::Result::Err(err) => {
                                          return ::core::result::Result::Err(::core::convert::From::from(err))
                                      }
                                  },})
                }
            }
            #[doc = r" Returns the module that was created."]
            #[allow(dead_code)]
            #[inline]
            pub fn module(&self)
             -> &::std::sync::Arc<::vulkano::pipeline::shader::ShaderModule> {
                &self.shader
            }
            #[doc =
                  r" Returns a logical struct describing the entry point named `{ep_name}`."]
            #[inline]
            #[allow(unsafe_code)]
            pub fn main_entry_point(&self)
             ->
                 ::vulkano::pipeline::shader::GraphicsEntryPoint<(),
                                                                 MainInput,
                                                                 MainOutput,
                                                                 Layout> {
                unsafe {
                    #[allow(dead_code)]
                    static NAME: [u8; 5usize] =
                        [109u8, 97u8, 105u8, 110u8, 0];
                    self.shader.graphics_entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr()
                                                                                    as
                                                                                    *const _),
                                                     MainInput, MainOutput,
                                                     Layout(ShaderStages{fragment:
                                                                             true,
                                                                                     ..ShaderStages::none()}),
                                                     ::vulkano::pipeline::shader::GraphicsShaderType::Fragment)
                }
            }
        }
        #[structural_match]
        #[rustc_copy_clone_marker]
        pub struct MainInput;
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::fmt::Debug for MainInput {
            fn fmt(&self, f: &mut ::std::fmt::Formatter)
             -> ::std::fmt::Result {
                match *self {
                    MainInput => {
                        let mut debug_trait_builder =
                            f.debug_tuple("MainInput");
                        debug_trait_builder.finish()
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::marker::Copy for MainInput { }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::clone::Clone for MainInput {
            #[inline]
            fn clone(&self) -> MainInput { { *self } }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::cmp::PartialEq for MainInput {
            #[inline]
            fn eq(&self, other: &MainInput) -> bool {
                match *other {
                    MainInput => match *self { MainInput => true, },
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::cmp::Eq for MainInput {
            #[inline]
            #[doc(hidden)]
            fn assert_receiver_is_total_eq(&self) -> () { { } }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::hash::Hash for MainInput {
            fn hash<__H: ::std::hash::Hasher>(&self, state: &mut __H) -> () {
                match *self { MainInput => { } }
            }
        }
        #[allow(unsafe_code)]
        unsafe impl ::vulkano::pipeline::shader::ShaderInterfaceDef for
         MainInput {
            type
            Iter
            =
            MainInputIter;
            fn elements(&self) -> MainInputIter { MainInputIter{num: 0,} }
        }
        #[rustc_copy_clone_marker]
        pub struct MainInputIter {
            num: u16,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::fmt::Debug for MainInputIter {
            fn fmt(&self, f: &mut ::std::fmt::Formatter)
             -> ::std::fmt::Result {
                match *self {
                    MainInputIter { num: ref __self_0_0 } => {
                        let mut debug_trait_builder =
                            f.debug_struct("MainInputIter");
                        let _ =
                            debug_trait_builder.field("num", &&(*__self_0_0));
                        debug_trait_builder.finish()
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::marker::Copy for MainInputIter { }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::clone::Clone for MainInputIter {
            #[inline]
            fn clone(&self) -> MainInputIter {
                { let _: ::std::clone::AssertParamIsClone<u16>; *self }
            }
        }
        impl Iterator for MainInputIter {
            type
            Item
            =
            ::vulkano::pipeline::shader::ShaderInterfaceDefEntry;
            #[inline]
            fn next(&mut self) -> Option<Self::Item> { None }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = 0usize - self.num as usize;
                (len, Some(len))
            }
        }
        impl ExactSizeIterator for MainInputIter { }
        #[structural_match]
        #[rustc_copy_clone_marker]
        pub struct MainOutput;
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::fmt::Debug for MainOutput {
            fn fmt(&self, f: &mut ::std::fmt::Formatter)
             -> ::std::fmt::Result {
                match *self {
                    MainOutput => {
                        let mut debug_trait_builder =
                            f.debug_tuple("MainOutput");
                        debug_trait_builder.finish()
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::marker::Copy for MainOutput { }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::clone::Clone for MainOutput {
            #[inline]
            fn clone(&self) -> MainOutput { { *self } }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::cmp::PartialEq for MainOutput {
            #[inline]
            fn eq(&self, other: &MainOutput) -> bool {
                match *other {
                    MainOutput => match *self { MainOutput => true, },
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::cmp::Eq for MainOutput {
            #[inline]
            #[doc(hidden)]
            fn assert_receiver_is_total_eq(&self) -> () { { } }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::hash::Hash for MainOutput {
            fn hash<__H: ::std::hash::Hasher>(&self, state: &mut __H) -> () {
                match *self { MainOutput => { } }
            }
        }
        #[allow(unsafe_code)]
        unsafe impl ::vulkano::pipeline::shader::ShaderInterfaceDef for
         MainOutput {
            type
            Iter
            =
            MainOutputIter;
            fn elements(&self) -> MainOutputIter { MainOutputIter{num: 0,} }
        }
        #[rustc_copy_clone_marker]
        pub struct MainOutputIter {
            num: u16,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::fmt::Debug for MainOutputIter {
            fn fmt(&self, f: &mut ::std::fmt::Formatter)
             -> ::std::fmt::Result {
                match *self {
                    MainOutputIter { num: ref __self_0_0 } => {
                        let mut debug_trait_builder =
                            f.debug_struct("MainOutputIter");
                        let _ =
                            debug_trait_builder.field("num", &&(*__self_0_0));
                        debug_trait_builder.finish()
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::marker::Copy for MainOutputIter { }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::clone::Clone for MainOutputIter {
            #[inline]
            fn clone(&self) -> MainOutputIter {
                { let _: ::std::clone::AssertParamIsClone<u16>; *self }
            }
        }
        impl Iterator for MainOutputIter {
            type
            Item
            =
            ::vulkano::pipeline::shader::ShaderInterfaceDefEntry;
            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                if self.num == 0u16 {
                    self.num += 1;
                    return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry{location:
                                                                                         0u32..1u32,
                                                                                     format:
                                                                                         ::vulkano::format::Format::R32G32B32A32Sfloat,
                                                                                     name:
                                                                                         Some(::std::borrow::Cow::Borrowed("f_color")),});
                }
                None
            }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = 1usize - self.num as usize;
                (len, Some(len))
            }
        }
        impl ExactSizeIterator for MainOutputIter { }
        pub mod ty { }
        pub struct Layout(pub ShaderStages);
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::fmt::Debug for Layout {
            fn fmt(&self, f: &mut ::std::fmt::Formatter)
             -> ::std::fmt::Result {
                match *self {
                    Layout(ref __self_0_0) => {
                        let mut debug_trait_builder = f.debug_tuple("Layout");
                        let _ = debug_trait_builder.field(&&(*__self_0_0));
                        debug_trait_builder.finish()
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::std::clone::Clone for Layout {
            #[inline]
            fn clone(&self) -> Layout {
                match *self {
                    Layout(ref __self_0_0) =>
                    Layout(::std::clone::Clone::clone(&(*__self_0_0))),
                }
            }
        }
        #[allow(unsafe_code)]
        unsafe impl PipelineLayoutDesc for Layout {
            fn num_sets(&self) -> usize { 0usize }
            fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
                match set { _ => None, }
            }
            fn descriptor(&self, set: usize, binding: usize)
             -> Option<DescriptorDesc> {
                match (set, binding) { _ => None, }
            }
            fn num_push_constants_ranges(&self) -> usize { 0usize }
            fn push_constants_range(&self, num: usize)
             -> Option<PipelineLayoutDescPcRange> {
                if num != 0 || 0usize == 0 {
                    None
                } else {
                    Some(PipelineLayoutDescPcRange{offset: 0,
                                                   size: 0usize,
                                                   stages:
                                                       ShaderStages::all(),})
                }
            }
        }
        #[allow(non_snake_case)]
        #[repr(C)]
        #[rustc_copy_clone_marker]
        pub struct SpecializationConstants {
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        #[allow(non_snake_case)]
        impl ::std::fmt::Debug for SpecializationConstants {
            fn fmt(&self, f: &mut ::std::fmt::Formatter)
             -> ::std::fmt::Result {
                match *self {
                    SpecializationConstants {  } => {
                        let mut debug_trait_builder =
                            f.debug_struct("SpecializationConstants");
                        debug_trait_builder.finish()
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        #[allow(non_snake_case)]
        impl ::std::marker::Copy for SpecializationConstants { }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        #[allow(non_snake_case)]
        impl ::std::clone::Clone for SpecializationConstants {
            #[inline]
            fn clone(&self) -> SpecializationConstants { { *self } }
        }
        impl Default for SpecializationConstants {
            fn default() -> SpecializationConstants {
                SpecializationConstants{}
            }
        }
        unsafe impl SpecConstsTrait for SpecializationConstants {
            fn descriptors() -> &'static [SpecializationMapEntry] {
                static DESCRIPTORS: [SpecializationMapEntry; 0usize] = [];
                &DESCRIPTORS
            }
        }
    }
    let _cube_vs = cube_vs::Shader::load(device.clone()).unwrap();
    let _cube_fs = cube_fs::Shader::load(device.clone()).unwrap();
    let render_pass =
        Arc::new({
                     use ::vulkano::framebuffer::RenderPassDesc;
                     mod scope {
                         #![allow(non_camel_case_types)]
                         #![allow(non_snake_case)]
                         use ::vulkano::format::ClearValue;
                         use ::vulkano::format::Format;
                         use ::vulkano::framebuffer::RenderPassDesc;
                         use ::vulkano::framebuffer::RenderPassDescClearValues;
                         use ::vulkano::framebuffer::AttachmentDescription;
                         use ::vulkano::framebuffer::PassDescription;
                         use ::vulkano::framebuffer::PassDependencyDescription;
                         use ::vulkano::image::ImageLayout;
                         use ::vulkano::sync::AccessFlagBits;
                         use ::vulkano::sync::PipelineStages;
                         pub struct CustomRenderPassDesc {
                             pub color: (Format, u32),
                             pub depth: (Format, u32),
                         }
                         #[allow(unsafe_code)]
                         unsafe impl RenderPassDesc for CustomRenderPassDesc {
                             #[inline]
                             fn num_attachments(&self) -> usize {
                                 num_attachments()
                             }
                             #[inline]
                             fn attachment_desc(&self, id: usize)
                              -> Option<AttachmentDescription> {
                                 attachment(self, id)
                             }
                             #[inline]
                             fn num_subpasses(&self) -> usize {
                                 num_subpasses()
                             }
                             #[inline]
                             fn subpass_desc(&self, id: usize)
                              -> Option<PassDescription> {
                                 subpass(id)
                             }
                             #[inline]
                             fn num_dependencies(&self) -> usize {
                                 num_dependencies()
                             }
                             #[inline]
                             fn dependency_desc(&self, id: usize)
                              -> Option<PassDependencyDescription> {
                                 dependency(id)
                             }
                         }
                         unsafe impl RenderPassDescClearValues<Vec<ClearValue>>
                          for CustomRenderPassDesc {
                             fn convert_clear_values(&self,
                                                     values: Vec<ClearValue>)
                              -> Box<Iterator<Item = ClearValue>> {
                                 Box::new(values.into_iter())
                             }
                         }
                         #[inline]
                         fn num_attachments() -> usize {
                             #![allow(unused_assignments)]
                             #![allow(unused_mut)]
                             #![allow(unused_variables)]
                             let mut num = 0;
                             let color = num;
                             num += 1;
                             let depth = num;
                             num += 1;
                             num
                         }
                         #[inline]
                         fn attachment(desc: &CustomRenderPassDesc, id: usize)
                          -> Option<AttachmentDescription> {
                             #![allow(unused_assignments)]
                             #![allow(unused_mut)]
                             let mut num = 0;
                             {
                                 if id == num {
                                     let (initial_layout, final_layout) =
                                         attachment_layouts(num);
                                     return Some(::vulkano::framebuffer::AttachmentDescription{format:
                                                                                                   desc.color.0,
                                                                                               samples:
                                                                                                   desc.color.1,
                                                                                               load:
                                                                                                   ::vulkano::framebuffer::LoadOp::Clear,
                                                                                               store:
                                                                                                   ::vulkano::framebuffer::StoreOp::Store,
                                                                                               stencil_load:
                                                                                                   ::vulkano::framebuffer::LoadOp::Clear,
                                                                                               stencil_store:
                                                                                                   ::vulkano::framebuffer::StoreOp::Store,
                                                                                               initial_layout:
                                                                                                   initial_layout,
                                                                                               final_layout:
                                                                                                   final_layout,});
                                 }
                                 num += 1;
                             }
                             {
                                 if id == num {
                                     let (initial_layout, final_layout) =
                                         attachment_layouts(num);
                                     return Some(::vulkano::framebuffer::AttachmentDescription{format:
                                                                                                   desc.depth.0,
                                                                                               samples:
                                                                                                   desc.depth.1,
                                                                                               load:
                                                                                                   ::vulkano::framebuffer::LoadOp::Clear,
                                                                                               store:
                                                                                                   ::vulkano::framebuffer::StoreOp::DontCare,
                                                                                               stencil_load:
                                                                                                   ::vulkano::framebuffer::LoadOp::Clear,
                                                                                               stencil_store:
                                                                                                   ::vulkano::framebuffer::StoreOp::DontCare,
                                                                                               initial_layout:
                                                                                                   initial_layout,
                                                                                               final_layout:
                                                                                                   final_layout,});
                                 }
                                 num += 1;
                             }
                             None
                         }
                         #[inline]
                         fn num_subpasses() -> usize {
                             #![allow(unused_assignments)]
                             #![allow(unused_mut)]
                             #![allow(unused_variables)]
                             let mut num = 0;
                             let color = num;
                             num += 1;
                             num
                         }
                         #[inline]
                         fn subpass(id: usize) -> Option<PassDescription> {
                             #![allow(unused_assignments)]
                             #![allow(unused_mut)]
                             #![allow(unused_variables)]
                             let mut attachment_num = 0;
                             let color = attachment_num;
                             attachment_num += 1;
                             let depth = attachment_num;
                             attachment_num += 1;
                             let mut cur_pass_num = 0;
                             {
                                 if id == cur_pass_num {
                                     let mut depth = None;
                                     depth =
                                         Some((depth,
                                               ImageLayout::DepthStencilAttachmentOptimal));
                                     let mut desc =
                                         PassDescription{color_attachments:
                                                             <[_]>::into_vec(box
                                                                                 [(color,
                                                                                   ImageLayout::ColorAttachmentOptimal)]),
                                                         depth_stencil: depth,
                                                         input_attachments:
                                                             <[_]>::into_vec(box
                                                                                 []),
                                                         resolve_attachments:
                                                             <[_]>::into_vec(box
                                                                                 []),
                                                         preserve_attachments:
                                                             (0..attachment_num).filter(|&a|
                                                                                            {
                                                                                                if a
                                                                                                       ==
                                                                                                       color
                                                                                                   {
                                                                                                    return false;
                                                                                                }
                                                                                                if a
                                                                                                       ==
                                                                                                       depth
                                                                                                   {
                                                                                                    return false;
                                                                                                }
                                                                                                true
                                                                                            }).collect(),};
                                     if !(desc.resolve_attachments.is_empty()
                                              ||
                                              desc.resolve_attachments.len()
                                                  ==
                                                  desc.color_attachments.len())
                                        {
                                         {
                                             ::std::rt::begin_panic("assertion failed: desc.resolve_attachments.is_empty() ||\n    desc.resolve_attachments.len() == desc.color_attachments.len()",
                                                                    &("src/main.rs",
                                                                      134u32,
                                                                      32u32))
                                         }
                                     };
                                     return Some(desc);
                                 }
                                 cur_pass_num += 1;
                             }
                             None
                         }
                         #[inline]
                         fn num_dependencies() -> usize {
                             num_subpasses().saturating_sub(1)
                         }
                         #[inline]
                         fn dependency(id: usize)
                          -> Option<PassDependencyDescription> {
                             let num_passes = num_subpasses();
                             if id + 1 >= num_passes { return None; }
                             Some(PassDependencyDescription{source_subpass:
                                                                id,
                                                            destination_subpass:
                                                                id + 1,
                                                            source_stages:
                                                                PipelineStages{all_graphics:
                                                                                   true,
                                                                                           ..PipelineStages::none()},
                                                            destination_stages:
                                                                PipelineStages{all_graphics:
                                                                                   true,
                                                                                           ..PipelineStages::none()},
                                                            source_access:
                                                                AccessFlagBits::all(),
                                                            destination_access:
                                                                AccessFlagBits::all(),
                                                            by_region: true,})
                         }
                         /// Returns the initial and final layout of an attachment, given its num.
                         ///
                         /// The value always correspond to the first and last usages of an attachment.
                         fn attachment_layouts(num: usize)
                          -> (ImageLayout, ImageLayout) {
                             #![allow(unused_assignments)]
                             #![allow(unused_mut)]
                             #![allow(unused_variables)]
                             let mut attachment_num = 0;
                             let color = attachment_num;
                             attachment_num += 1;
                             let depth = attachment_num;
                             attachment_num += 1;
                             let mut initial_layout = None;
                             let mut final_layout = None;
                             {
                                 if depth == num {
                                     if initial_layout.is_none() {
                                         initial_layout =
                                             Some(ImageLayout::DepthStencilAttachmentOptimal);
                                     }
                                     final_layout =
                                         Some(ImageLayout::DepthStencilAttachmentOptimal);
                                 }
                                 if color == num {
                                     if initial_layout.is_none() {
                                         initial_layout =
                                             Some(ImageLayout::ColorAttachmentOptimal);
                                     }
                                     final_layout =
                                         Some(ImageLayout::ColorAttachmentOptimal);
                                 }
                             }
                             if color == num {
                                 if initial_layout ==
                                        Some(ImageLayout::DepthStencilAttachmentOptimal)
                                        ||
                                        initial_layout ==
                                            Some(ImageLayout::ColorAttachmentOptimal)
                                        ||
                                        initial_layout ==
                                            Some(ImageLayout::TransferDstOptimal)
                                    {
                                     if ::vulkano::framebuffer::LoadOp::Clear
                                            ==
                                            ::vulkano::framebuffer::LoadOp::Clear
                                            ||
                                            ::vulkano::framebuffer::LoadOp::Clear
                                                ==
                                                ::vulkano::framebuffer::LoadOp::DontCare
                                        {
                                         initial_layout =
                                             Some(ImageLayout::Undefined);
                                     }
                                 }
                             }
                             if depth == num {
                                 if initial_layout ==
                                        Some(ImageLayout::DepthStencilAttachmentOptimal)
                                        ||
                                        initial_layout ==
                                            Some(ImageLayout::ColorAttachmentOptimal)
                                        ||
                                        initial_layout ==
                                            Some(ImageLayout::TransferDstOptimal)
                                    {
                                     if ::vulkano::framebuffer::LoadOp::Clear
                                            ==
                                            ::vulkano::framebuffer::LoadOp::Clear
                                            ||
                                            ::vulkano::framebuffer::LoadOp::Clear
                                                ==
                                                ::vulkano::framebuffer::LoadOp::DontCare
                                        {
                                         initial_layout =
                                             Some(ImageLayout::Undefined);
                                     }
                                 }
                             }
                             (initial_layout.expect(::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["Attachment ",
                                                                                                         " is missing initial_layout, this is normally automatically determined but you can manually specify it for an individual attachment in the single_pass_renderpass! macro"],
                                                                                                       &match (&attachment_num,)
                                                                                                            {
                                                                                                            (arg0,)
                                                                                                            =>
                                                                                                            [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                                         ::std::fmt::Display::fmt)],
                                                                                                        })).as_ref()),
                              final_layout.expect(::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["Attachment ",
                                                                                                       " is missing final_layout, this is normally automatically determined but you can manually specify it for an individual attachment in the single_pass_renderpass! macro"],
                                                                                                     &match (&attachment_num,)
                                                                                                          {
                                                                                                          (arg0,)
                                                                                                          =>
                                                                                                          [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                                       ::std::fmt::Display::fmt)],
                                                                                                      })).as_ref()))
                         }
                     }
                     scope::CustomRenderPassDesc{color:
                                                     (swapchain.format(), 1),
                                                 depth:
                                                     (Format::D16Unorm,
                                                      1),}.build_render_pass(device.clone())
                 }.unwrap());
    let teapot_pipeline =
        Arc::new(GraphicsPipeline::start().vertex_input_single_buffer::<Vertex>().vertex_shader(teapot_vs.main_entry_point(),
                                                                                                ()).triangle_list().depth_stencil_simple_depth().viewports_dynamic_scissors_irrelevant(1).blend_alpha_blending().fragment_shader(teapot_fs.main_entry_point(),
                                                                                                                                                                                                                                 ()).render_pass(Subpass::from(render_pass.clone(),
                                                                                                                                                                                                                                                               0).unwrap()).build(device.clone()).unwrap());
    let sampler =
        Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
                     MipmapMode::Nearest, SamplerAddressMode::Repeat,
                     SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
                     0.0, 1.0, 0.0, 0.0).unwrap();
    let (_document, _buffers, images) =
        gltf::import("9_mm/scene.gltf").unwrap();
    for mesh in _document.meshes() {
        if mesh.name() == Some("Plane001_Plane_0") { break ; }
        match mesh.name() {
            tmp => {
                {
                    ::std::io::_eprint(::std::fmt::Arguments::new_v1_formatted(&["[",
                                                                                 ":",
                                                                                 "] ",
                                                                                 " = ",
                                                                                 "\n"],
                                                                               &match (&"src/main.rs",
                                                                                       &185u32,
                                                                                       &"mesh.name()",
                                                                                       &&tmp)
                                                                                    {
                                                                                    (arg0,
                                                                                     arg1,
                                                                                     arg2,
                                                                                     arg3)
                                                                                    =>
                                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                 ::std::fmt::Display::fmt),
                                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                                 ::std::fmt::Display::fmt),
                                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                                 ::std::fmt::Display::fmt),
                                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                                 ::std::fmt::Debug::fmt)],
                                                                                },
                                                                               &[::std::fmt::rt::v1::Argument{position:
                                                                                                                  ::std::fmt::rt::v1::Position::At(0usize),
                                                                                                              format:
                                                                                                                  ::std::fmt::rt::v1::FormatSpec{fill:
                                                                                                                                                     ' ',
                                                                                                                                                 align:
                                                                                                                                                     ::std::fmt::rt::v1::Alignment::Unknown,
                                                                                                                                                 flags:
                                                                                                                                                     0u32,
                                                                                                                                                 precision:
                                                                                                                                                     ::std::fmt::rt::v1::Count::Implied,
                                                                                                                                                 width:
                                                                                                                                                     ::std::fmt::rt::v1::Count::Implied,},},
                                                                                 ::std::fmt::rt::v1::Argument{position:
                                                                                                                  ::std::fmt::rt::v1::Position::At(1usize),
                                                                                                              format:
                                                                                                                  ::std::fmt::rt::v1::FormatSpec{fill:
                                                                                                                                                     ' ',
                                                                                                                                                 align:
                                                                                                                                                     ::std::fmt::rt::v1::Alignment::Unknown,
                                                                                                                                                 flags:
                                                                                                                                                     0u32,
                                                                                                                                                 precision:
                                                                                                                                                     ::std::fmt::rt::v1::Count::Implied,
                                                                                                                                                 width:
                                                                                                                                                     ::std::fmt::rt::v1::Count::Implied,},},
                                                                                 ::std::fmt::rt::v1::Argument{position:
                                                                                                                  ::std::fmt::rt::v1::Position::At(2usize),
                                                                                                              format:
                                                                                                                  ::std::fmt::rt::v1::FormatSpec{fill:
                                                                                                                                                     ' ',
                                                                                                                                                 align:
                                                                                                                                                     ::std::fmt::rt::v1::Alignment::Unknown,
                                                                                                                                                 flags:
                                                                                                                                                     0u32,
                                                                                                                                                 precision:
                                                                                                                                                     ::std::fmt::rt::v1::Count::Implied,
                                                                                                                                                 width:
                                                                                                                                                     ::std::fmt::rt::v1::Count::Implied,},},
                                                                                 ::std::fmt::rt::v1::Argument{position:
                                                                                                                  ::std::fmt::rt::v1::Position::At(3usize),
                                                                                                              format:
                                                                                                                  ::std::fmt::rt::v1::FormatSpec{fill:
                                                                                                                                                     ' ',
                                                                                                                                                 align:
                                                                                                                                                     ::std::fmt::rt::v1::Alignment::Unknown,
                                                                                                                                                 flags:
                                                                                                                                                     4u32,
                                                                                                                                                 precision:
                                                                                                                                                     ::std::fmt::rt::v1::Count::Implied,
                                                                                                                                                 width:
                                                                                                                                                     ::std::fmt::rt::v1::Count::Implied,},}]));
                };
                tmp
            }
        };
        for primitive in mesh.primitives() {
            let mut normal_tex_coords = <[_]>::into_vec(box []);
            let reader =
                primitive.reader(|buffer| Some(&_buffers[buffer.index()]));
            let mut descriptor_sets_collection =
                components::graphics::DescriptorSetsCollection::default();
            if let (Some(color), Some(roughness), Some(emissive),
                    Some(normal)) =
                   (primitive.material().pbr_metallic_roughness().base_color_texture(),
                    primitive.material().pbr_metallic_roughness().metallic_roughness_texture(),
                    primitive.material().emissive_texture(),
                    primitive.material().normal_texture()) {
                match reader.read_tex_coords(0) {
                    Some(gltf::mesh::util::ReadTexCoords::F32(t)) =>
                    normal_tex_coords.extend(t),
                    _ => { }
                }
                let mut color_texture = &images[color.texture().index()];
                use vulkano::sync::GpuFuture;
                let color_image =
                    ImmutableImage::from_iter(color_texture.pixels.clone().into_iter(),
                                              Dimensions::Dim2d{width:
                                                                    color_texture.width,
                                                                height:
                                                                    color_texture.height,},
                                              Format::R8G8B8A8Srgb,
                                              queue.clone()).unwrap();
                (color_image.1).then_signal_fence_and_flush().unwrap().wait(None).unwrap();
                let roughness_texture = &images[roughness.texture().index()];
                {
                    ::std::io::_print(::std::fmt::Arguments::new_v1(&["",
                                                                      "\n"],
                                                                    &match (&roughness_texture.format,)
                                                                         {
                                                                         (arg0,)
                                                                         =>
                                                                         [::std::fmt::ArgumentV1::new(arg0,
                                                                                                      ::std::fmt::Debug::fmt)],
                                                                     }));
                };
                let pixels: Vec<u8> =
                    roughness_texture.pixels.clone().into_iter().tuples().map(|(a,
                                                                                b,
                                                                                c)|
                                                                                  <[_]>::into_vec(box
                                                                                                      [a,
                                                                                                       b,
                                                                                                       c,
                                                                                                       std::u8::MAX])).flatten().collect();
                let roughness_image =
                    ImmutableImage::from_iter(pixels.into_iter(),
                                              Dimensions::Dim2d{width:
                                                                    roughness_texture.width,
                                                                height:
                                                                    roughness_texture.height,},
                                              Format::R8G8B8A8Srgb,
                                              queue.clone()).unwrap();
                let emissive_texture = &images[emissive.texture().index()];
                let pixels: Vec<u8> =
                    emissive_texture.pixels.clone().into_iter().tuples().map(|(a,
                                                                               b,
                                                                               c)|
                                                                                 <[_]>::into_vec(box
                                                                                                     [a,
                                                                                                      b,
                                                                                                      c,
                                                                                                      std::u8::MAX])).flatten().collect();
                let emissive_image =
                    ImmutableImage::from_iter(pixels.into_iter(),
                                              Dimensions::Dim2d{width:
                                                                    emissive_texture.width,
                                                                height:
                                                                    emissive_texture.height,},
                                              Format::R8G8B8A8Srgb,
                                              queue.clone()).unwrap();
                let normal_texture = &images[normal.texture().index()];
                let pixels: Vec<u8> =
                    normal_texture.pixels.clone().into_iter().tuples().map(|(a,
                                                                             b,
                                                                             c)|
                                                                               <[_]>::into_vec(box
                                                                                                   [a,
                                                                                                    b,
                                                                                                    c,
                                                                                                    std::u8::MAX])).flatten().collect();
                let normal_image =
                    ImmutableImage::from_iter(pixels.into_iter(),
                                              Dimensions::Dim2d{width:
                                                                    normal_texture.width,
                                                                height:
                                                                    normal_texture.height,},
                                              Format::R8G8B8A8Srgb,
                                              queue.clone()).unwrap();
                let texture_descriptor =
                    PersistentDescriptorSet::start(teapot_pipeline.clone(),
                                                   0).add_sampled_image(color_image.0.clone(),
                                                                        sampler.clone()).unwrap().add_sampled_image(roughness_image.0.clone(),
                                                                                                                    sampler.clone()).unwrap().add_sampled_image(emissive_image.0.clone(),
                                                                                                                                                                sampler.clone()).unwrap().add_sampled_image(normal_image.0.clone(),
                                                                                                                                                                                                            sampler.clone()).unwrap().build().unwrap();
                descriptor_sets_collection.push_or_replace(0,
                                                           Arc::new(texture_descriptor));
            } else {
                {
                    ::std::io::_print(::std::fmt::Arguments::new_v1(&["WARNING! NO TEXTURE\n"],
                                                                    &match ()
                                                                         {
                                                                         () =>
                                                                         [],
                                                                     }));
                };
            }
            let mut vertices: Vec<Vertex> = <[_]>::into_vec(box []);
            for (position, normals, tex_coords) in
                ::itertools::__std_iter::IntoIterator::into_iter(reader.read_positions().unwrap()).zip(reader.read_normals().unwrap()).zip(normal_tex_coords).map(|((a,
                                                                                                                                                                     b),
                                                                                                                                                                    b)|
                                                                                                                                                                      (a,
                                                                                                                                                                       b,
                                                                                                                                                                       b))
                {
                vertices.push(Vertex{position, normals, tex_coords,});
            }
            let mesh: Mesh<Vertex, u32> =
                Mesh{indices:
                         match reader.read_indices().unwrap() {
                             gltf::mesh::util::ReadIndices::U32(iter) =>
                             iter.collect(),
                             _ => <[_]>::into_vec(box []),
                         },
                     vertices,};
            let immutable_indices_buf =
                CpuAccessibleBuffer::from_iter(device.clone(),
                                               BufferUsage::all(),
                                               mesh.indices.into_iter()).unwrap();
            let immutable_vert_buf =
                CpuAccessibleBuffer::from_iter(device.clone(),
                                               BufferUsage::all(),
                                               mesh.vertices.into_iter()).unwrap();
            world.create_entity().with(MeshBuffer::<Vertex,
                                                    u32>::from(<[_]>::into_vec(box
                                                                                   [immutable_vert_buf.clone()]),
                                                               immutable_indices_buf.clone())).with(components::graphics::GraphicsPipeline(teapot_pipeline.clone())).with(components::graphics::FixedSizeDescriptorSetsPool(FixedSizeDescriptorSetsPool::new(teapot_pipeline.clone(),
                                                                                                                                                                                                                                                             1))).with(Isometry::<f32>::default()).with(descriptor_sets_collection).build();
        }
    }
    let (models, _materials) = load_obj(&Path::new("./teapot.obj")).unwrap();
    use frozengame::model::Vertex;
    let mut physics_world: PhysicsWorld<f32> = PhysicsWorld::new();
    let shape =
        ShapeHandle::new(Cuboid::new(nalgebra::Vector3::new(0.5, 1.0, 0.5)));
    let collider =
        ColliderDesc::new(shape).density(1.3).material(MaterialHandle::new(BasicMaterial::new(1.2,
                                                                                              0.8))).margin(0.02);
    let rigid_body =
        RigidBodyDesc::new().name("player body".to_owned()).collider(&collider).mass(2.2).velocity(Velocity3::linear(0.0,
                                                                                                                     0.0,
                                                                                                                     0.0)).build(&mut physics_world);
    let self_player =
        world.create_entity().with(RigidBody(rigid_body.handle())).build();
    world.add_resource(ActivePlayer(self_player));
    let mut dynamic_state = DynamicState::default();
    let framebuffers =
        window_size_dependent_setup(device.clone(), &swapchain_images,
                                    render_pass.clone(), &mut dynamic_state);
    world.add_resource(physics_world);
    world.add_resource::<Option<Arc<Swapchain<winit::Window>>>>(Some(swapchain.clone()));
    world.add_resource::<Vec<Arc<FramebufferAbstract + Send +
                                 Sync>>>(framebuffers);
    world.add_resource::<Arc<Device>>(device.clone());
    world.add_resource::<Option<Arc<Queue>>>(Some(queue.clone()));
    let (image, init) =
        ImmutableImage::uninitialized(device.clone(),
                                      Dimensions::Dim2d{width: 200,
                                                        height: 200,},
                                      swapchain.format(), MipmapsCount::One,
                                      ImageUsage{sampled:
                                                     true,
                                                             ..ImageUsage::none()},
                                      ImageLayout::General,
                                      device.active_queue_families()).unwrap();
    let mut dispatcher =
        DispatcherBuilder::new().with(PhysicsSystem::<f32>::default(),
                                      "physics system",
                                      &[]).with(EventSystem, "event system",
                                                &[]).with(MeshUniformSystem,
                                                          "mesh system",
                                                          &[]).with(RenderSystem{previous_frame:
                                                                                     None,},
                                                                    "render system",
                                                                    &["mesh system"]).build();
    world.add_resource(dynamic_state);
    dispatcher.setup(&mut world.res);
    dispatcher.dispatch(&mut world.res);
    world.maintain();
    let mut events = Vec::new();
    world.add_resource(events.clone());
    loop  {
        events.clear();
        if let Ok(ref mut events_loop) = events_loop.write() {
            events_loop.poll_events(|event| events.push(event));
        }
        world.add_resource(events.clone());
        dispatcher.dispatch(&world.res);
    }
}
fn window_size_dependent_setup(device: Arc<Device>,
                               images: &[Arc<SwapchainImage<Window>>],
                               render_pass:
                                   Arc<RenderPassAbstract + Send + Sync>,
                               dynamic_state: &mut DynamicState)
 -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();
    let depth_buffer =
        AttachmentImage::transient(device.clone(), dimensions.width_height(),
                                   Format::D16Unorm).unwrap();
    let viewport =
        Viewport{origin: [0.0, 0.0],
                 dimensions:
                     [dimensions.width() as f32, dimensions.height() as f32],
                 depth_range: 0.0..1.0,};
    dynamic_state.viewports = Some(<[_]>::into_vec(box [viewport]));
    images.iter().map(|image|
                          {
                              Arc::new(Framebuffer::start(render_pass.clone()).add(image.clone()).unwrap().add(depth_buffer.clone()).unwrap().build().unwrap())
                                  as Arc<FramebufferAbstract + Send + Sync>
                          }).collect::<Vec<_>>()
}
