use anyhow::Result;
use nvml_wrapper::Nvml;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

static MAX_LOADING_THRESHOLD: u32 = 90;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceUtilization {
    pub gpu_util: u32,
    pub memory_util: u32,
    pub memory_used: u64,
    pub memory_total: u64,
}

pub struct LocalGPUManager {
    pub nvml: Nvml,
    pub device_count: u32,
}

impl LocalGPUManager {
    pub fn try_new() -> Result<Self> {
        let nvml = Nvml::init()?;
        let device_count = nvml.device_count()?;

        Ok(LocalGPUManager { nvml, device_count })
    }

    pub fn is_overloaded(&self) -> Result<bool> {
        let utilization_data = self.get_gpu_utilization()?;

        for (_, util) in utilization_data.iter() {
            if util.gpu_util > MAX_LOADING_THRESHOLD || util.memory_util > MAX_LOADING_THRESHOLD {
                return Ok(true);
            }
        }

        Ok(false)
    }

    pub fn get_gpu_utilization(&self) -> Result<HashMap<String, DeviceUtilization>> {
        let mut result = HashMap::new();

        for i in 0..self.device_count {
            let device = self.nvml.device_by_index(i)?;
            let name = device.name()?;
            let utilization = device.utilization_rates()?;
            let memory = device.memory_info()?;

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

        Ok(result)
    }

    pub fn print_utilization(&self) -> Result<()> {
        let utilization_data = self.get_gpu_utilization()?;

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
