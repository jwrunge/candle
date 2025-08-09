fn get_cpu() -> candle_core::Device {
    println!("Failed to initialize Metal device, falling back to CPU");
    candle_core::Device::Cpu
}

pub fn get() -> candle_core::Device {
    #[cfg(target_os = "macos")]
    {
        candle_core::Device::new_metal(0).unwrap_or_else(|_| get_cpu())
    }
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        candle_core::Device::new_cuda(0).unwrap_or_else(|_| get_cpu())
    }
}
