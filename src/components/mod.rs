use specs::{Component, VecStorage, System, ReadStorage, Read, WriteStorage, Write, Resources, WriteExpect, ReadExpect, Entities};
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

pub mod player;
pub mod physics;
pub mod movement;
pub mod graphics;

#[derive(Default)]
pub struct PressedKeys(pub Vec<VirtualKeyCode>);

pub struct MovementSystem;

impl Component for Camera<f32> {
    type Storage = VecStorage<Self>;
}
