use anyhow::Result;
use nvml_wrapper::Nvml;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeviceUtilization {
    gpu_util: u32,
    memory_util: u32,
    memory_used: u64,
    memory_total: u64,
}

struct LocalGPUManager {
    pub nvml: Nvml,
    pub device_count: u32,
}

impl LocalGPUManager {
    fn try_new() -> Result<Self> {
        let nvml = Nvml::init()?;
        let device_count = nvml.device_count()?;

        Ok(LocalGPUManager { nvml, device_count })
    }

    fn get_gpu_utilization(&self) -> Result<HashMap<String, DeviceUtilization>> {
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

    fn print_utilization(&self) -> Result<()> {
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
