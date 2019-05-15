use specs::{System, WriteExpect, WriteStorage};
use alga::general::RealField;
use std::marker::PhantomData;
use nphysics3d::world::World as PhysicsWorld;
use crate::components::physics::{Shape, RigidBody};
use specs::join::Join;
use std::time::{Duration, Instant};

pub struct PhysicsSystem<N>(PhantomData<N>, pub Instant);

impl<N> Default for PhysicsSystem<N> {
    fn default() -> Self {
        PhysicsSystem(PhantomData::default(), Instant::now())
    }
}

impl<'a, N: RealField> System<'a> for PhysicsSystem<N> {
    type SystemData = (
        WriteExpect<'a, PhysicsWorld<N>>,
        WriteStorage<'a, RigidBody>,
    );

    fn run(&mut self, (mut physics_world, rigid_bodies): Self::SystemData) {
        if Instant::now().duration_since(self.1).as_secs_f64() >= physics_world.timestep().to_subset().unwrap() {
//            physics_world.colliders().for_each(|c| {
//                println!("{:?}", c.position());
//            });
            physics_world.step();
            self.1 = Instant::now();
        }
    }

}