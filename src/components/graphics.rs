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

pub struct MeshBuffer<VD, ID>(pub Vec<Arc<BufferAccess + Send + Sync + 'static>>, pub Arc<TypedBufferAccess<Content = [ID]> + Send + Sync + 'static>, PhantomData<VD>);

impl<VD, ID> MeshBuffer<VD, ID> {
    pub fn from(v: Vec<Arc<BufferAccess + Send + Sync + 'static>>, i: Arc<TypedBufferAccess<Content = [ID]> + Send + Sync + 'static>) -> Self {
        MeshBuffer(v, i, PhantomData::default())
    }
}

impl<VD, ID> Component for MeshBuffer<VD, ID>
    where
        MeshBuffer<VD, ID>: Send + Sync + 'static
{
    type Storage = VecStorage<Self>;
}

pub struct Mesh<VD, IT>(pub frozengame::model::Mesh<VD, IT>);


impl Component for Mesh<Vertex, u32> {
    type Storage = VecStorage<Self>;
}

pub struct Texture(Sampler, ImmutableImage<Format>);

impl Component for Texture
{
    type Storage = FlaggedStorage<Self, VecStorage<Self>>;
}

pub struct GraphicsPipeline(pub Arc<GraphicsPipelineAbstract + Send + Sync + 'static>);

impl Component for GraphicsPipeline {
    type Storage = VecStorage<Self>;
}

pub struct FixedSizeDescriptorSetsPool(pub descriptor_set::FixedSizeDescriptorSetsPool<Arc<PipelineLayoutAbstract + Send + Sync>>);

impl Component for FixedSizeDescriptorSetsPool
{
    type Storage = VecStorage<Self>;
}

pub struct DescriptorSetsCollection(pub Vec<Arc<DescriptorSet + Send + Sync>>);

impl DescriptorSetsCollection {
    pub fn push_or_replace(&mut self, index: usize, descriptor_set: Arc<DescriptorSet + Send + Sync>) {
        if self.0.len() <= index {
            self.0.insert(index, descriptor_set);
        } else {
            self.0[index] = descriptor_set;
        }
    }
}

impl Default for DescriptorSetsCollection {
    fn default() -> Self {
        DescriptorSetsCollection(Vec::default())
    }
}

impl Component for DescriptorSetsCollection {
    type Storage = VecStorage<Self>;
}
