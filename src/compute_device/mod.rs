pub fn get() -> candle_core::Device {
    #[cfg(target_os = "macos")]
    {
        candle_core::Device::new_metal(0).expect("Failed to create Metal device")
    }
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        candle_core::Device::new_cuda(0).expect("Failed to create CUDA device")
    }
}
