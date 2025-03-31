use anyhow::{Context, Result};
use nvml_wrapper::{error::NvmlError, Nvml};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

static MAX_LOADING_THRESHOLD: u32 = 90;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceUtilization {
    pub gpu_util: u32,
    pub memory_util: u32,
    pub memory_used: u64,
    pub memory_total: u64,
}

pub struct LocalGPUManager {
    pub nvml: Option<Nvml>,
    pub device_count: u32,
}

impl LocalGPUManager {
    pub fn new() -> Self {
        match Nvml::init() {
            Ok(nvml) => {
                let device_count = nvml.device_count().unwrap_or(0);
                LocalGPUManager {
                    nvml: Some(nvml),
                    device_count,
                }
            }
            Err(_) => {
                warn!(
                    "Could not initialize NVML. No NVIDIA GPUs detected. Continuing without GPU."
                );
                LocalGPUManager {
                    nvml: None,
                    device_count: 0,
                }
            }
        }
    }

    pub fn is_overloaded(&self) -> Result<bool> {
        if !self.has_nvidia_gpu()? {
            return Ok(false);
        }

        let utilization_data = self.get_gpu_utilization()?;

        for (_, util) in utilization_data.iter() {
            if util.gpu_util > MAX_LOADING_THRESHOLD || util.memory_util > MAX_LOADING_THRESHOLD {
                return Ok(true);
            }
        }

        Ok(false)
    }

    pub fn has_nvidia_gpu(&self) -> Result<bool> {
        Ok(self.nvml.is_some() && self.device_count > 0)
    }

    pub fn get_gpu_utilization(&self) -> Result<HashMap<String, DeviceUtilization>> {
        let mut result = HashMap::new();

        if let Some(nvml) = &self.nvml {
            for i in 0..self.device_count {
                match nvml.device_by_index(i) {
                    Ok(device) => {
                        let name = device.name().context("Failed to get device name")?;
                        let utilization = device
                            .utilization_rates()
                            .context("Failed to get utilization rates")?;
                        let memory = device.memory_info().context("Failed to get memory info")?;

                        result.insert(
                            format!("{} (ID: {})", name, i),
                            DeviceUtilization {
                                gpu_util: utilization.gpu,
                                memory_util: utilization.memory,
                                memory_used: memory.used,
                                memory_total: memory.total,
                            },
                        );
                    }
                    Err(e) => {
                        warn!(
                            "Error accessing device {}: {}. Execution will continue without GPUs.",
                            i, e
                        );
                        continue;
                    }
                }
            }
        }

        Ok(result)
    }

    pub fn print_utilization(&self) -> Result<()> {
        if !self.has_nvidia_gpu()? {
            info!("No NVIDIA GPUs detected on this system. Continuing without GPU.");
            return Ok(());
        }

        let utilization_data = self.get_gpu_utilization()?;

        if utilization_data.is_empty() {
            warn!("GPU information could not be retrieved.");
            return Ok(());
        }

        println!("=== GPU Utilization ===");
        for (device_name, util) in utilization_data.iter() {
            println!("Device: {}", device_name);
            println!("  GPU Utilization: {}%", util.gpu_util);
            println!("  Memory Utilization: {}%", util.memory_util);
            println!(
                "  Memory Used: {:.2} MB / {:.2} MB",
                util.memory_used as f64 / 1024.0 / 1024.0,
                util.memory_total as f64 / 1024.0 / 1024.0
            );
            println!();
        }

        Ok(())
    }
}
