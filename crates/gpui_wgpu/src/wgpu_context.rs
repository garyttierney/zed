#[cfg(not(target_family = "wasm"))]
use anyhow::Context as _;
#[cfg(not(target_family = "wasm"))]
use gpui_util::ResultExt;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct WgpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    dual_source_blending: bool,
    device_lost: Arc<AtomicBool>,
}

#[derive(Clone, Copy)]
pub struct CompositorGpuHint {
    pub vendor_id: u32,
    pub device_id: u32,
}

/// Callback that can filter or modify the set of wgpu features before device
/// creation. Receives the adapter info and the candidate feature set, and
/// returns the feature set that should actually be requested.
pub type FeatureFilterFn =
    Box<dyn Fn(&wgpu::AdapterInfo, wgpu::Features) -> wgpu::Features + 'static>;

/// Configuration for GPU context initialization.
///
/// Use the builder methods to customise which wgpu features and limits are
/// requested when the device is created.
pub struct GpuContextConfig {
    additional_features: wgpu::Features,
    all_adapter_features: bool,
    experimental_features: bool,
    adapter_limits: bool,
    feature_filter: Option<FeatureFilterFn>,
}

impl Default for GpuContextConfig {
    fn default() -> Self {
        Self {
            additional_features: wgpu::Features::empty(),
            all_adapter_features: false,
            experimental_features: false,
            adapter_limits: false,
            feature_filter: None,
        }
    }
}

impl GpuContextConfig {
    /// Request additional wgpu features on top of what GPUI requires internally.
    pub fn with_additional_features(mut self, features: wgpu::Features) -> Self {
        self.additional_features = features;
        self
    }

    /// When `true`, request *all* features reported by the adapter instead of
    /// only the ones GPUI needs.
    pub fn with_all_adapter_features(mut self, enabled: bool) -> Self {
        self.all_adapter_features = enabled;
        self
    }

    /// When `true`, enable experimental wgpu features on the device descriptor.
    ///
    /// # Safety
    /// Experimental features may contain UB-triggering bugs. See
    /// [`wgpu::ExperimentalFeatures::enabled`] for details.
    pub fn with_experimental_features(mut self, enabled: bool) -> Self {
        self.experimental_features = enabled;
        self
    }

    /// When `true`, request limits that match the adapter's reported limits
    /// rather than the conservative `downlevel_defaults`.
    pub fn with_adapter_limits(mut self, enabled: bool) -> Self {
        self.adapter_limits = enabled;
        self
    }

    /// Register a callback that is invoked just before device creation to
    /// adjust the requested feature set. This runs after all other feature
    /// selection logic.
    pub fn with_feature_filter(
        mut self,
        filter: impl Fn(&wgpu::AdapterInfo, wgpu::Features) -> wgpu::Features + 'static,
    ) -> Self {
        self.feature_filter = Some(Box::new(filter));
        self
    }
}

struct GpuContextInner {
    context: Option<WgpuContext>,
    config: GpuContextConfig,
}

/// Shared GPU context reference, used to coordinate device recovery across
/// multiple windows and to carry configuration for device creation.
#[derive(Clone)]
pub struct GpuContext {
    inner: Rc<RefCell<GpuContextInner>>,
}

impl GpuContext {
    /// Create a new `GpuContext` with the given configuration.
    pub fn new(config: GpuContextConfig) -> Self {
        Self {
            inner: Rc::new(RefCell::new(GpuContextInner {
                context: None,
                config,
            })),
        }
    }

    /// Create a `GpuContext` that already holds an initialised `WgpuContext`.
    ///
    /// The config is set to defaults since the device was created externally.
    pub fn from_wgpu_context(wgpu_context: WgpuContext) -> Self {
        Self {
            inner: Rc::new(RefCell::new(GpuContextInner {
                context: Some(wgpu_context),
                config: GpuContextConfig::default(),
            })),
        }
    }

    /// Returns `true` if the underlying `WgpuContext` has been initialised and
    /// reports that the GPU device was lost.
    pub fn device_lost(&self) -> bool {
        self.inner
            .borrow()
            .context
            .as_ref()
            .is_some_and(|ctx| ctx.device_lost())
    }

    /// Returns `true` if the `WgpuContext` has not been initialised, or was
    /// initialised but the device has been lost.
    pub fn needs_new_context(&self) -> bool {
        self.inner
            .borrow()
            .context
            .as_ref()
            .is_none_or(|ctx| ctx.device_lost())
    }

    /// Returns `true` if dual-source blending is supported by the current GPU.
    pub fn supports_dual_source_blending(&self) -> bool {
        self.inner
            .borrow()
            .context
            .as_ref()
            .is_some_and(|ctx| ctx.supports_dual_source_blending())
    }

    /// Clone the wgpu `Instance` from the underlying context, if initialised.
    pub fn instance(&self) -> Option<wgpu::Instance> {
        self.inner
            .borrow()
            .context
            .as_ref()
            .map(|ctx| ctx.instance.clone())
    }

    /// Run a callback with read access to the `WgpuContext`, if initialised.
    pub fn with_context<R>(&self, f: impl FnOnce(&WgpuContext) -> R) -> Option<R> {
        let inner = self.inner.borrow();
        inner.context.as_ref().map(f)
    }

    /// Check that the current adapter is compatible with `surface`.
    pub fn check_compatible_with_surface(&self, surface: &wgpu::Surface<'_>) -> anyhow::Result<()> {
        let inner = self.inner.borrow();
        match inner.context.as_ref() {
            Some(ctx) => ctx.check_compatible_with_surface(surface),
            None => Ok(()),
        }
    }

    /// Initialise the `WgpuContext` if it has not been created yet.
    ///
    /// If the context already exists this is a no-op.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn ensure_initialized(
        &self,
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
        compositor_gpu: Option<CompositorGpuHint>,
    ) -> anyhow::Result<()> {
        let mut inner = self.inner.borrow_mut();
        if inner.context.is_none() {
            let wgpu_ctx = WgpuContext::new(instance, surface, compositor_gpu, &inner.config)?;
            inner.context = Some(wgpu_ctx);
        }
        Ok(())
    }

    /// Replace the current `WgpuContext` with a freshly-created one.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn replace_context(
        &self,
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
        compositor_gpu: Option<CompositorGpuHint>,
    ) -> anyhow::Result<()> {
        let mut inner = self.inner.borrow_mut();
        let new_context = WgpuContext::new(instance, surface, compositor_gpu, &inner.config)?;
        inner.context = Some(new_context);
        Ok(())
    }

    /// Drop the current `WgpuContext`, if any.
    pub(crate) fn clear_context(&self) {
        self.inner.borrow_mut().context.take();
    }

    /// Run a callback with read access to the underlying `WgpuContext`.
    ///
    /// Panics if the context has not been initialised.
    pub(crate) fn with_context_ref<R>(&self, f: impl FnOnce(&WgpuContext) -> R) -> R {
        let inner = self.inner.borrow();
        f(inner.context.as_ref().expect("WgpuContext not initialised"))
    }
}

impl WgpuContext {
    #[cfg(not(target_family = "wasm"))]
    pub fn new(
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
        compositor_gpu: Option<CompositorGpuHint>,
        config: &GpuContextConfig,
    ) -> anyhow::Result<Self> {
        let device_id_filter = match std::env::var("ZED_DEVICE_ID") {
            Ok(val) => parse_pci_id(&val)
                .context("Failed to parse device ID from `ZED_DEVICE_ID` environment variable")
                .log_err(),
            Err(std::env::VarError::NotPresent) => None,
            err => {
                err.context("Failed to read value of `ZED_DEVICE_ID` environment variable")
                    .log_err();
                None
            }
        };

        // Select an adapter by actually testing surface configuration with the real device.
        // This is the only reliable way to determine compatibility on hybrid GPU systems.
        let (adapter, device, queue, dual_source_blending) =
            pollster::block_on(Self::select_adapter_and_device(
                &instance,
                device_id_filter,
                surface,
                compositor_gpu.as_ref(),
                config,
            ))?;

        let device_lost = Arc::new(AtomicBool::new(false));
        device.set_device_lost_callback({
            let device_lost = Arc::clone(&device_lost);
            move |reason, message| {
                log::error!("wgpu device lost: reason={reason:?}, message={message}");
                if reason != wgpu::DeviceLostReason::Destroyed {
                    device_lost.store(true, Ordering::Relaxed);
                }
            }
        });

        log::info!(
            "Selected GPU adapter: {:?} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            dual_source_blending,
            device_lost,
        })
    }

    /// Creates a `WgpuContext` from externally-initialized wgpu resources.
    ///
    /// Use this when you need to share a wgpu `Instance`, `Adapter`, `Device`,
    /// and `Queue` with other wgpu-based libraries (e.g. a video decoder or
    /// compute framework). The caller is responsible for creating the resources
    /// and ensuring the adapter and device are compatible with the surfaces that
    /// GPUI will create for its windows.
    ///
    /// A device-lost callback is installed on the provided device; if the caller
    /// has already set one it will be replaced.
    pub fn from_shared_resources(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Self {
        let dual_source_blending = adapter
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING);

        let device_lost = Arc::new(AtomicBool::new(false));
        device.set_device_lost_callback({
            let device_lost = Arc::clone(&device_lost);
            move |reason, message| {
                log::error!("wgpu device lost: reason={reason:?}, message={message}");
                if reason != wgpu::DeviceLostReason::Destroyed {
                    device_lost.store(true, Ordering::Relaxed);
                }
            }
        });

        Self {
            instance,
            adapter,
            device,
            queue,
            dual_source_blending,
            device_lost,
        }
    }

    #[cfg(target_family = "wasm")]
    pub async fn new_web() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to request GPU adapter: {e}"))?;

        log::info!(
            "Selected GPU adapter: {:?} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        let device_lost = Arc::new(AtomicBool::new(false));
        let (device, queue, dual_source_blending) =
            Self::create_device(&adapter, &GpuContextConfig::default()).await?;

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            dual_source_blending,
            device_lost,
        })
    }

    /// Request a wgpu device from `adapter` with the features and limits that
    /// GPUI's renderer requires (e.g. dual-source blending when available),
    /// plus any extras specified by `config`.
    ///
    /// Returns `(device, queue, dual_source_blending)`.
    pub async fn create_device(
        adapter: &wgpu::Adapter,
        config: &GpuContextConfig,
    ) -> anyhow::Result<(wgpu::Device, wgpu::Queue, bool)> {
        let dual_source_blending = adapter
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING);

        let mut required_features = wgpu::Features::empty();
        if dual_source_blending {
            required_features |= wgpu::Features::DUAL_SOURCE_BLENDING;
        } else {
            log::warn!(
                "Dual-source blending not available on this GPU. \
                Subpixel text antialiasing will be disabled."
            );
        }

        if config.all_adapter_features {
            required_features |= adapter.features();
        }

        required_features |= config.additional_features;

        if let Some(ref filter) = config.feature_filter {
            required_features = filter(&adapter.get_info(), required_features);
        }

        let required_limits = if config.adapter_limits {
            adapter.limits()
        } else {
            wgpu::Limits::downlevel_defaults()
                .using_resolution(adapter.limits())
                .using_alignment(adapter.limits())
        };

        // Safety: the caller opted in via `GpuContextConfig::with_experimental_features(true)`.
        let experimental_features = if config.experimental_features {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gpui_device"),
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                experimental_features,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create wgpu device: {e}"))?;

        Ok((device, queue, dual_source_blending))
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn instance(display: Box<dyn wgpu::wgt::WgpuHasDisplayHandle>) -> wgpu::Instance {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: Some(display),
        })
    }

    pub fn check_compatible_with_surface(&self, surface: &wgpu::Surface<'_>) -> anyhow::Result<()> {
        let caps = surface.get_capabilities(&self.adapter);
        if caps.formats.is_empty() {
            let info = self.adapter.get_info();
            anyhow::bail!(
                "Adapter {:?} (backend={:?}, device={:#06x}) is not compatible with the \
                 display surface for this window.",
                info.name,
                info.backend,
                info.device,
            );
        }
        Ok(())
    }

    /// Select an adapter and create a device, testing that the surface can actually be configured.
    /// This is the only reliable way to determine compatibility on hybrid GPU systems, where
    /// adapters may report surface compatibility via get_capabilities() but fail when actually
    /// configuring (e.g., NVIDIA reporting Vulkan Wayland support but failing because the
    /// Wayland compositor runs on the Intel GPU).
    #[cfg(not(target_family = "wasm"))]
    async fn select_adapter_and_device(
        instance: &wgpu::Instance,
        device_id_filter: Option<u32>,
        surface: &wgpu::Surface<'_>,
        compositor_gpu: Option<&CompositorGpuHint>,
        config: &GpuContextConfig,
    ) -> anyhow::Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue, bool)> {
        let mut adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).await;

        if adapters.is_empty() {
            anyhow::bail!("No GPU adapters found");
        }

        if let Some(device_id) = device_id_filter {
            log::info!("ZED_DEVICE_ID filter: {:#06x}", device_id);
        }

        // Sort adapters into a single priority order. Tiers (from highest to lowest):
        //
        // 1. ZED_DEVICE_ID match — explicit user override
        // 2. Compositor GPU match — the GPU the display server is rendering on
        // 3. Device type (Discrete > Integrated > Other > Virtual > Cpu).
        //    "Other" ranks above "Virtual" because OpenGL seems to count as "Other".
        // 4. Backend — prefer Vulkan/Metal/Dx12 over GL/etc.
        adapters.sort_by_key(|adapter| {
            let info = adapter.get_info();

            // Backends like OpenGL report device=0 for all adapters, so
            // device-based matching is only meaningful when non-zero.
            let device_known = info.device != 0;

            let user_override: u8 = match device_id_filter {
                Some(id) if device_known && info.device == id => 0,
                _ => 1,
            };

            let compositor_match: u8 = match compositor_gpu {
                Some(hint)
                    if device_known
                        && info.vendor == hint.vendor_id
                        && info.device == hint.device_id =>
                {
                    0
                }
                _ => 1,
            };

            let type_priority: u8 = match info.device_type {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                wgpu::DeviceType::Other => 2,
                wgpu::DeviceType::VirtualGpu => 3,
                wgpu::DeviceType::Cpu => 4,
            };

            let backend_priority: u8 = match info.backend {
                wgpu::Backend::Vulkan => 0,
                wgpu::Backend::Metal => 0,
                wgpu::Backend::Dx12 => 0,
                _ => 1,
            };

            (
                user_override,
                compositor_match,
                type_priority,
                backend_priority,
            )
        });

        // Log all available adapters (in sorted order)
        log::info!("Found {} GPU adapter(s):", adapters.len());
        for adapter in &adapters {
            let info = adapter.get_info();
            log::info!(
                "  - {} (vendor={:#06x}, device={:#06x}, backend={:?}, type={:?})",
                info.name,
                info.vendor,
                info.device,
                info.backend,
                info.device_type,
            );
        }

        // Test each adapter by creating a device and configuring the surface
        for adapter in adapters {
            let info = adapter.get_info();
            log::info!("Testing adapter: {} ({:?})...", info.name, info.backend);

            match Self::try_adapter_with_surface(&adapter, surface, config).await {
                Ok((device, queue, dual_source_blending)) => {
                    log::info!(
                        "Selected GPU (passed configuration test): {} ({:?})",
                        info.name,
                        info.backend
                    );
                    return Ok((adapter, device, queue, dual_source_blending));
                }
                Err(e) => {
                    log::info!(
                        "  Adapter {} ({:?}) failed: {}, trying next...",
                        info.name,
                        info.backend,
                        e
                    );
                }
            }
        }

        anyhow::bail!("No GPU adapter found that can configure the display surface")
    }

    /// Try to use an adapter with a surface by creating a device and testing configuration.
    /// Returns the device and queue if successful, allowing them to be reused.
    #[cfg(not(target_family = "wasm"))]
    async fn try_adapter_with_surface(
        adapter: &wgpu::Adapter,
        surface: &wgpu::Surface<'_>,
        config: &GpuContextConfig,
    ) -> anyhow::Result<(wgpu::Device, wgpu::Queue, bool)> {
        let caps = surface.get_capabilities(adapter);
        if caps.formats.is_empty() {
            anyhow::bail!("no compatible surface formats");
        }
        if caps.alpha_modes.is_empty() {
            anyhow::bail!("no compatible alpha modes");
        }

        let (device, queue, dual_source_blending) = Self::create_device(adapter, config).await?;
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let test_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width: 64,
            height: 64,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &test_config);

        let error = error_scope.pop().await;
        if let Some(e) = error {
            anyhow::bail!("surface configuration failed: {e}");
        }

        Ok((device, queue, dual_source_blending))
    }

    pub fn supports_dual_source_blending(&self) -> bool {
        self.dual_source_blending
    }

    /// Returns true if the GPU device was lost (e.g., due to driver crash, suspend/resume).
    /// When this returns true, the context should be recreated.
    pub fn device_lost(&self) -> bool {
        self.device_lost.load(Ordering::Relaxed)
    }

    /// Returns a clone of the device_lost flag for sharing with renderers.
    pub(crate) fn device_lost_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.device_lost)
    }
}

#[cfg(not(target_family = "wasm"))]
fn parse_pci_id(id: &str) -> anyhow::Result<u32> {
    let mut id = id.trim();

    if id.starts_with("0x") || id.starts_with("0X") {
        id = &id[2..];
    }
    let is_hex_string = id.chars().all(|c| c.is_ascii_hexdigit());
    let is_4_chars = id.len() == 4;
    anyhow::ensure!(
        is_4_chars && is_hex_string,
        "Expected a 4 digit PCI ID in hexadecimal format"
    );

    u32::from_str_radix(id, 16).context("parsing PCI ID as hex")
}

#[cfg(test)]
mod tests {
    use super::parse_pci_id;

    #[test]
    fn test_parse_device_id() {
        assert!(parse_pci_id("0xABCD").is_ok());
        assert!(parse_pci_id("ABCD").is_ok());
        assert!(parse_pci_id("abcd").is_ok());
        assert!(parse_pci_id("1234").is_ok());
        assert!(parse_pci_id("123").is_err());
        assert_eq!(
            parse_pci_id(&format!("{:x}", 0x1234)).unwrap(),
            parse_pci_id(&format!("{:X}", 0x1234)).unwrap(),
        );

        assert_eq!(
            parse_pci_id(&format!("{:#x}", 0x1234)).unwrap(),
            parse_pci_id(&format!("{:#X}", 0x1234)).unwrap(),
        );
    }
}
