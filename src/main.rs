use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::device::{Device, Features, DeviceExtensions};
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage};
use winit::{EventsLoop, WindowBuilder};
use vulkano_win::VkSurfaceBuild;
use vulkano::swapchain::{Swapchain, SurfaceTransform, PresentMode};

fn main() {
    let extensions = vulkano_win::required_extensions();

    let instance = Instance::new(None, &extensions, None).expect("No vulkan available");

    // Take the first physical device
    let physical_device = PhysicalDevice::enumerate(&instance).next().expect("Couldn't find device");
    println!("{:?}", physical_device.name());

    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();
    let window = surface.window();


    // Find a queue family that supports graphical usage.
    let queue_family = physical_device.queue_families()
        .find(|&q| q.supports_graphics() && q.supports_compute() && q.supports_transfers() && surface.is_supported(q).unwrap_or(false))
        .expect("couldn't find a queue supporting graphics");

    println!("Queues: {:?}", queue_family.queues_count());

    let device_extensions = DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::none() };

    let (device, mut queues) = {
        Device::new(
            physical_device,
            &Features::none(),
            &device_extensions,
            [(queue_family, 0.5)].iter().cloned()
        ).expect("")
    };

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical_device).unwrap();

        let usage = caps.supported_usage_flags;

        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        let format = caps.supported_formats[0].0;

        let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else {
            return;
        };

        Swapchain::new(
            device.clone(), surface.clone(), caps.min_image_count, format, initial_dimensions, 1,
            usage, &queue, SurfaceTransform::Identity, alpha, PresentMode::Fifo, true, None
        ).unwrap()
    };


    loop {

    }

}