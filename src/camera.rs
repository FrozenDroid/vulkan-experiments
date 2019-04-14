use cgmath::{Array, Deg, Vector3, Point3, Quaternion, BaseNum, Rotation, Rotation3, RelativeEq, BaseFloat, InnerSpace, Matrix4, EuclideanSpace, SquareMatrix};
use core::borrow::Borrow;

#[derive(Debug)]
pub struct Camera<T> {
    pitch:      Deg<T>,
    heading:    Deg<T>,
    up:         Vector3<T>,
    look_at:    Vector3<T>,
    position:   Vector3<T>,
    direction:  Vector3<T>,
    view_matrix: Matrix4<T>,
}

impl<T> Camera<T>
    where
        T: BaseFloat,
        T: RelativeEq,
{

    pub fn pitch(&self) -> &Deg<T> {
        &self.pitch
    }

    pub fn set_pitch<A: Into<Deg<T>>>(&mut self, pitch: A) {
        self.pitch = pitch.into();
        self.update();
    }

    pub fn heading(&self) -> &Deg<T> {
        &self.heading
    }

    pub fn set_heading<A: Into<Deg<T>>>(&mut self, heading: A) {
        self.heading = heading.into();
        self.update();
    }

    pub fn position(&self) -> &Vector3<T> {
        &self.position
    }

    pub fn set_position(&mut self, position: Vector3<T>) {
        self.position = position;
        self.update();
    }

    fn update(&mut self) {
        self.direction = (self.look_at - self.position).normalize();
        let mut pitch_axis = self.direction.cross(self.up);
        let mut pitch_quat = Quaternion::from_axis_angle(pitch_axis, self.pitch);
        let mut heading_quat = Quaternion::from_axis_angle(self.up, self.heading);

        let mut temp_quat = ((pitch_quat * heading_quat) as Quaternion<T>).normalize();
        self.direction = temp_quat.rotate_vector(self.direction);
        self.look_at = self.position + self.direction;

        self.view_matrix = Matrix4::look_at(
            Point3::from_vec(self.position),
            Point3::from_vec(self.look_at),
            self.up
        );
    }

    pub fn view_matrix(&self) -> Matrix4<T> {
        self.view_matrix
    }

}

impl Default for Camera<f32> {
    fn default() -> Self {
        Camera {
            pitch:      Deg(0.0),
            heading:    Deg(0.0),
            up:         Vector3::new(0.0, 1.0, 0.0),
            look_at:    Vector3::new(0.0, 0.0, 0.0),
            position:   Vector3::new(0.0, 0.0, 0.0),
            direction:  Vector3::new(0.0, 0.0, 0.0),
            view_matrix: Matrix4::identity(),
        }
    }
}