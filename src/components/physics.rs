use specs::{Component, VecStorage};
use alga::general::RealField;
use ncollide3d::shape::Shape as PhysicsShape;
use nphysics3d::math::Force as PhysicsForce;
use std::sync::Arc;
use nphysics3d::object::{RigidBody as PhysicsRigidBody, BodyHandle};

pub struct Shape<N: RealField>(Arc<PhysicsShape<N>>);

impl<N: RealField> Component for Shape<N> {
    type Storage = VecStorage<Self>;
}

pub struct RigidBody(pub BodyHandle);

impl Component for RigidBody {
    type Storage = VecStorage<Self>;
}

pub struct Force<N: RealField>(Arc<PhysicsForce<N>>);

impl<N: RealField> Component for Force<N> {
    type Storage = VecStorage<Self>;
}
