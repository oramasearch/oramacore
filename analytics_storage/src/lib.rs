use anyhow::{anyhow, Context, Result};
use ms_converter::ms;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub type TimeSeriesData<T> = Vec<(i64, T)>;
pub type BlockSpan = (i64, i64);

pub enum Granularity {
    Hour,
    Day,
    Week,
    Month,
}

pub enum OffloadTarget {
    Void,
    S3,
    R2,
}

#[derive(Default)]
pub struct AnalyticsStorageConfig {
    pub index_id: String,
    pub buffer_size: Option<usize>,
    pub granularity: Option<Granularity>,
    pub persistence_dir: Option<String>,
    pub offload_after: Option<usize>,
    pub offload_to: Option<OffloadTarget>,
}

pub struct AnalyticsStorage<T: Serialize + Clone> {
    index_id: String,
    docs_size: usize,
    buffer_size: usize,
    granularity: Granularity,
    persistence_dir: String,
    offload_after: usize,
    offload_to: OffloadTarget,
    write_buffer: VecDeque<T>,
}

pub enum Version {
    V1_0,
}

#[derive(Serialize, Clone)]
pub struct VersionV1_0Schema {
    pub id: String,
    pub deployment_id: String,
    pub instance_id: String,
    pub timestamp: i64,
    pub raw_search_string: String,
    pub raw_query: String,
    pub results_count: usize,
    pub referer: String,
    pub pop: String,
    pub country: String,
    pub continent: String,
    pub visitor_id: String,
}

pub struct TimeBlockMeta {
    doc_size: usize,
    version: Version,
    path: String,
    size: u64,
}

pub struct TimeBlock<T> {
    meta: TimeBlockMeta,
    data: TimeSeriesData<T>,
}

const ORAMA_ANALYTICS_DEFAULT_DIRNAME: &str = ".orama_analytics";
const DEFAULT_BUFFER_SIZE: usize = 100;
const DEFAULT_OFFLOAD_AFTER: usize = 1440; // 60 days in hours

impl<T: Serialize + Clone> AnalyticsStorage<T> {
    pub fn try_new(config: AnalyticsStorageConfig) -> Result<Self> {
        let storage = AnalyticsStorage {
            index_id: config.index_id,
            docs_size: 0,
            buffer_size: config.buffer_size.unwrap_or(DEFAULT_BUFFER_SIZE),
            offload_to: config.offload_to.unwrap_or(OffloadTarget::Void),
            granularity: config.granularity.unwrap_or(Granularity::Month),
            offload_after: config.offload_after.unwrap_or(DEFAULT_OFFLOAD_AFTER),
            persistence_dir: config
                .persistence_dir
                .unwrap_or_else(|| ORAMA_ANALYTICS_DEFAULT_DIRNAME.to_string()),
            write_buffer: VecDeque::new(),
        };

        let index_dir = storage.get_index_dir();
        let index_path = Path::new(&index_dir);

        if !index_path.is_dir() {
            fs::create_dir_all(index_path)
                .with_context(|| format!("Failed to create directory {}", index_dir))?;
        }

        if !storage.current_block_exists()? {
            storage.create_new_block_file()?;
        }

        Ok(storage)
    }

    pub fn insert(&mut self, data: T) -> Result<()> {
        self.write_buffer.push_back(data);

        if self.write_buffer.len() >= self.buffer_size {
            let tmp = std::mem::take(&mut self.write_buffer);
            self.flush_buffer(tmp)?;
        }

        Ok(())
    }

    pub fn get_block_meta(path: &str) -> Result<TimeBlockMeta> {
        let file = fs::File::open(path).with_context(|| anyhow!("Unable to open file {}", path))?;
        let file_size = file.metadata()?.len();

        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        if let Some(first_line) = lines.next() {
            let line = first_line
                .with_context(|| anyhow!("Unable to read first line of file {}", path))?;
            let headers: Vec<&str> = line.split(',').collect();

            if headers.len() < 2 {
                return Err(anyhow!("Invalid header format in file {}", path));
            }

            let version = Self::parse_version(headers[0])?;
            let docs_size = headers[1]
                .parse::<usize>()
                .with_context(|| anyhow!("Invalid doc_size in header of file {}", path))?;

            return Ok(TimeBlockMeta {
                version,
                doc_size: docs_size,
                size: file_size,
                path: path.to_string(),
            });
        }

        Err(anyhow!("Cannot find header for time block file {}", path))
    }

    fn get_block(&self, timestamp: i64) -> Result<Option<TimeBlockMeta>> {
        let block_path = self.get_block_path(Some(timestamp))?;

        match block_path {
            Some(path) => Ok(Self::get_block_meta(&path).ok()),
            None => Ok(None),
        }
    }

    fn get_block_span(&self, timestamp: i64) -> Result<BlockSpan> {
        let duration: i64 = match self.granularity {
            Granularity::Hour => 3600,     // 1h in seconds
            Granularity::Day => 86400,     // 24h in seconds
            Granularity::Week => 604800,   // 7 days in seconds
            Granularity::Month => 2592000, // 30 days in seconds
        };

        Ok((timestamp - duration, timestamp + duration))
    }

    fn current_block_exists(&self) -> Result<bool> {
        let path = self.get_block_path(None)?;
        match path {
            Some(p) => Ok(Path::new(&p).exists()),
            None => Ok(false),
        }
    }

    fn get_block_path(&self, timestamp: Option<i64>) -> Result<Option<String>> {
        let now = timestamp.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("WTF Time went backwards")
                .as_millis() as i64
        });

        let (start, end) = self.get_block_span(now)?;
        let index_dir = self.get_index_dir();

        let result = fs::read_dir(&index_dir)?
            .filter_map(|res| res.ok())
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().to_string();
                name.parse::<i64>().ok()
            })
            .find(|&timestamp| timestamp >= start && timestamp <= end);

        Ok(result.map(|ts| format!("{}/{}", index_dir, ts)))
    }

    fn parse_version(version: &str) -> Result<Version> {
        match version {
            "1.0" => Ok(Version::V1_0),
            _ => Err(anyhow!("Invalid time block version number: {}", version)),
        }
    }

    fn create_new_block_file(&self) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis()
            .to_string();
        let file_path = format!("{}/{}", self.get_index_dir(), now);
        let mut file = File::create(file_path)?;
        writeln!(file, "1.0,0")?;
        file.sync_all()?;
        Ok(())
    }

    fn get_index_dir(&self) -> String {
        format!("{}/{}", self.persistence_dir, self.index_id)
    }

    fn flush_buffer(&mut self, mut tmp: VecDeque<T>) -> Result<()> {
        let mut retry_vec: Vec<_> = Vec::new();

        if !self.current_block_exists()? {
            self.create_new_block_file()?;
        }

        let current_block = self
            .get_block_path(None)?
            .with_context(|| "Unable to get the current block path while flushing")?;

        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(&current_block)?;

        while let Some(item) = tmp.pop_front() {
            let (timestamp, line) = self.format_line_for_block(item.clone())?;
            if let Err(e) = writeln!(file, "{},{}", timestamp, line) {
                eprintln!("Error writing to file: {:?}", e);
                retry_vec.push(item);
            }
        }

        // @todo: handle retries

        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if !self.write_buffer.is_empty() {
            let tmp = std::mem::take(&mut self.write_buffer);
            self.flush_buffer(tmp)?;
        }
        Ok(())
    }

    fn format_line_for_block(&self, data: T) -> Result<(i64, String)> {
        let mut wtr = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(Vec::new());
        wtr.serialize(&data)?;
        let str = String::from_utf8(wtr.into_inner()?)?.trim_end().to_string();
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as i64;
        Ok((now, str))
    }
}

impl<T: Serialize + Clone> Drop for AnalyticsStorage<T> {
    fn drop(&mut self) {
        if !self.write_buffer.is_empty() {
            let tmp = std::mem::take(&mut self.write_buffer);
            if let Err(e) = self.flush_buffer(tmp) {
                eprintln!("Error flushing buffer during drop: {:?}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup() -> (AnalyticsStorage<VersionV1_0Schema>, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = AnalyticsStorageConfig {
            index_id: "test_index".to_string(),
            persistence_dir: Some(temp_dir.path().to_str().unwrap().to_string()),
            granularity: Some(Granularity::Hour),
            ..Default::default()
        };
        let storage = AnalyticsStorage::try_new(config).unwrap();
        (storage, temp_dir)
    }

    #[test]
    fn test_block_span_hour() {
        let (storage, _temp) = setup();
        let timestamp = 3600; // 1 hour in seconds
        let (start, end) = storage.get_block_span(timestamp).unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, 7200);
    }

    #[test]
    fn test_block_span_day() {
        let config = AnalyticsStorageConfig {
            index_id: "test_index".to_string(),
            granularity: Some(Granularity::Day),
            ..Default::default()
        };
        let storage: AnalyticsStorage<VersionV1_0Schema> =
            AnalyticsStorage::try_new(config).unwrap();
        let timestamp = 86400; // 24 hours in seconds
        let (start, end) = storage.get_block_span(timestamp).unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, 172800);
    }

    #[test]
    fn test_insert_and_flush() {
        let (mut storage, _temp_dir) = setup();
        let test_data = VersionV1_0Schema {
            id: "test".to_string(),
            deployment_id: "test".to_string(),
            instance_id: "test".to_string(),
            timestamp: 1000,
            raw_search_string: "test".to_string(),
            raw_query: "test".to_string(),
            results_count: 0,
            referer: "test".to_string(),
            pop: "test".to_string(),
            country: "test".to_string(),
            continent: "test".to_string(),
            visitor_id: "test".to_string(),
        };

        for _ in 0..101 {
            storage.insert(test_data.clone()).unwrap();
        }

        let index_dir = storage.get_index_dir();
        let files: Vec<_> = fs::read_dir(index_dir)
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert!(!files.is_empty());
    }

    #[test]
    fn test_insert_and_verify_content() {
        let (mut storage, _temp_dir) = setup();
        let test_data = VersionV1_0Schema {
            id: "test".to_string(),
            deployment_id: "test".to_string(),
            instance_id: "test".to_string(),
            timestamp: 1000,
            raw_search_string: "test".to_string(),
            raw_query: "test".to_string(),
            results_count: 0,
            referer: "test".to_string(),
            pop: "test".to_string(),
            country: "test".to_string(),
            continent: "test".to_string(),
            visitor_id: "test".to_string(),
        };

        storage.insert(test_data.clone()).unwrap();

        let tmp = std::mem::take(&mut storage.write_buffer);
        storage.flush_buffer(tmp).unwrap();

        let block_path = storage.get_block_path(None).unwrap().unwrap();
        let file = File::open(block_path).unwrap();
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

        assert!(!lines.is_empty());
        assert_eq!(lines[0], "1.0,0");
        assert!(lines[1].contains("test"));
    }
}
