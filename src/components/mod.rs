use cgmath::Vector3;
use specs::{Component, VecStorage};

#[derive(Debug)]
struct Position(Vector3<f32>);

impl Component for Position {
    type Storage = VecStorage<Self>;
}
