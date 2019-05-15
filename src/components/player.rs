use specs::{Entity, Component, VecStorage};
use crate::components::movement::Isometry;
use alga::general::RealField;

pub struct ActivePlayer(pub Entity);
