mod dispatcher;
mod headless;
mod keyboard;
mod platform;
#[cfg(any(feature = "wayland", feature = "x11"))]
mod text_system;
#[cfg(feature = "wayland")]
mod wayland;
#[cfg(feature = "x11")]
mod x11;

#[cfg(any(feature = "wayland", feature = "x11"))]
mod xdg_desktop_portal;

pub use dispatcher::*;
pub(crate) use headless::*;
pub(crate) use keyboard::*;
pub(crate) use platform::*;
#[cfg(any(feature = "wayland", feature = "x11"))]
pub(crate) use text_system::*;
#[cfg(feature = "wayland")]
pub(crate) use wayland::*;
#[cfg(feature = "x11")]
pub(crate) use x11::*;

#[cfg(any(feature = "wayland", feature = "x11"))]
use std::cell::RefCell;
use std::rc::Rc;

/// Returns the default platform implementation for the current OS.
pub fn current_platform(headless: bool) -> Rc<dyn gpui::Platform> {
    #[cfg(feature = "x11")]
    use anyhow::Context as _;

    if headless {
        return Rc::new(LinuxPlatform {
            inner: HeadlessClient::new(),
        });
    }

    #[cfg(any(feature = "wayland", feature = "x11"))]
    let gpu_context: gpui_wgpu::GpuContext = Rc::new(RefCell::new(None));

    match gpui::guess_compositor() {
        #[cfg(feature = "wayland")]
        "Wayland" => Rc::new(LinuxPlatform {
            inner: WaylandClient::new(gpu_context),
        }),

        #[cfg(feature = "x11")]
        "X11" => Rc::new(LinuxPlatform {
            inner: X11Client::new(gpu_context)
                .context("Failed to initialize X11 client.")
                .unwrap(),
        }),

        "Headless" => Rc::new(LinuxPlatform {
            inner: HeadlessClient::new(),
        }),
        _ => unreachable!(),
    }
}

/// Returns a platform that shares the given wgpu resources with its renderer.
///
/// Use this when you need to share a wgpu `Instance`, `Adapter`, `Device`,
/// and `Queue` with external wgpu-based libraries.
#[cfg(any(feature = "wayland", feature = "x11"))]
pub fn current_platform_with_gpu(gpu_context: gpui_wgpu::GpuContext) -> Rc<dyn gpui::Platform> {
    #[cfg(feature = "x11")]
    use anyhow::Context as _;

    match gpui::guess_compositor() {
        #[cfg(feature = "wayland")]
        "Wayland" => Rc::new(LinuxPlatform {
            inner: WaylandClient::new(gpu_context),
        }),

        #[cfg(feature = "x11")]
        "X11" => Rc::new(LinuxPlatform {
            inner: X11Client::new(gpu_context)
                .context("Failed to initialize X11 client.")
                .unwrap(),
        }),

        "Headless" => Rc::new(LinuxPlatform {
            inner: HeadlessClient::new(),
        }),
        _ => unreachable!(),
    }
}
