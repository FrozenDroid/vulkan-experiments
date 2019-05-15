use nalgebra::{Isometry3, Isometry};
use alga::general::RealField;


#[derive(Debug)]
pub struct Camera<N: RealField> {
    isometry: Isometry3<N>,
}

impl<T> Camera<T>
    where
        T: RealField,
{

}

impl<N: RealField> Default for Camera<N> {
    fn default() -> Self {
        Camera {
            isometry: Isometry3::new(nalgebra::zero(), nalgebra::zero())
        }
    }
}
