use nalgebra::{RealField, Isometry3};
use specs::{Component, VecStorage};

impl<N: RealField> From<Isometry3<N>> for Isometry<N> {
    fn from(isometry: Isometry3<N>) -> Self {
        Isometry(isometry)
    }
}

#[derive(Debug)]
pub struct Isometry<N: RealField>(pub Isometry3<N>);

impl<N: RealField> Default for Isometry<N> {
    fn default() -> Self {
        Isometry(Isometry3::translation(nalgebra::zero(), nalgebra::zero(), nalgebra::zero()))
    }
}

impl<N: RealField> Component for Isometry<N> {
    type Storage = VecStorage<Self>;
}
