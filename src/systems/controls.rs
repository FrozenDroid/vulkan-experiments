use specs::{System, Read, ReadExpect, ReadStorage, WriteExpect, WriteStorage};
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

impl<'a> System<'a> for EventSystem {
    type SystemData = (
        Read<'a, Vec<Event>>,
        ReadExpect<'a, ActivePlayer>,
        WriteStorage<'a, RigidBody>,
        WriteExpect<'a, World<f32>>,
    );

    fn run(&mut self, (events, active_player, mut rigid_bodies, mut physics_world): Self::SystemData) {
        let mut rigid_body = rigid_bodies.get_mut(active_player.0).expect("expected player isometry");
        let mut rigid_body = physics_world.rigid_body_mut(rigid_body.0.clone()).unwrap();

        for event in events.clone().into_iter() {
            match event {
                Event::WindowEvent { event, .. } => {
                    match event {
                        WindowEvent::KeyboardInput { input, .. } => {
                            match input.virtual_keycode {
                                Some(VirtualKeyCode::W) => { rigid_body.apply_force(0, &Force3::linear(Vector3::new(0.0, 0.0, 100.0)), ForceType::Force, true); },
                                Some(VirtualKeyCode::S) => { rigid_body.apply_force(0, &Force3::linear(Vector3::new(0.0, 0.0, -100.0)), ForceType::Force, true); },
                                Some(VirtualKeyCode::A) => { rigid_body.apply_force(0, &Force3::linear(Vector3::new(-100.0, 0.0, 0.0)), ForceType::Force, true); } ,
                                Some(VirtualKeyCode::D) => { rigid_body.apply_force(0, &Force3::linear(Vector3::new(100.0, 0.0, 0.0)), ForceType::Force, true); },
                                _ => {}
                            };
                        },
                        _ => {}
                    };
                },
                Event::DeviceEvent { event, .. } => {
                    match event {
                        winit::DeviceEvent::MouseMotion { delta } => {
                            rigid_body.set_position(UnitQuaternion::from_axis_angle(&(rigid_body.position().rotation * Vector3::y_axis()), (-delta.0 as f32) * 0.01) * rigid_body.position());
                            rigid_body.set_position(UnitQuaternion::from_axis_angle(&(Vector3::x_axis()), (delta.1 as f32) * 0.01) * rigid_body.position());
                        },
                        _ => {},
                    }
                }
                _ => {}
            };
        }

//        match events.deref() {
//            Some(ev) => {
//
//            },
//            _ => {}
//        };
    }
}