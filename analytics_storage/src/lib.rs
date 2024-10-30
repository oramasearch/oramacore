use anyhow::{anyhow, Context, Result};
use ms_converter::ms;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

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

pub struct AnalyticsStorage<T> {
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

#[derive(Serialize)]
pub struct VersionV1_0Schema {
    id: String,
    deployment_id: String,
    instance_id: String,
    timestamp: i64,
    raw_search_string: String,
    raw_query: String,
    results_count: usize,
    referer: String,
    pop: String,
    country: String,
    continent: String,
    visitor_id: String,
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
        match self.granularity {
            Granularity::Hour => {
                let hour = ms("1 hour")?;
                Ok((timestamp - hour, timestamp + hour))
            }
            Granularity::Day => {
                let day = ms("1 day")?;
                Ok((timestamp - day, timestamp + day))
            }
            Granularity::Week => {
                let week = ms("1 week")?;
                Ok((timestamp - week, timestamp + week))
            }
            Granularity::Month => {
                let month = ms("30 days")?;
                Ok((timestamp - month, timestamp + month))
            }
        }
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
                .expect("Time went backwards")
                .as_millis() as i64
        });

        let lifespan = self.get_block_span(now)?;
        let index_dir = self.get_index_dir();

        let result = fs::read_dir(&index_dir)
            .with_context(|| format!("Failed to read directory {}", index_dir))?
            .filter_map(|res| res.ok())
            .map(|entry| entry.path().display().to_string())
            .find(|file_path| {
                file_path
                    .parse::<i64>()
                    .map(|time| time > lifespan.0 && time < lifespan.1)
                    .unwrap_or(false)
            });

        match result {
            Some(path) => Ok(Some(path)),
            None => Ok(None),
        }
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
        let file = File::create(file_path);
        file?.write_all(b"1.0,0")?;

        Ok(())
    }

    fn get_index_dir(&self) -> String {
        format!("{}/{}", self.persistence_dir, self.index_id)
    }

    fn flush_buffer(&mut self, mut tmp: VecDeque<T>) -> Result<()> {
        let mut retry_vec: Vec<_> = Vec::new();
        let current_block = self
            .get_block_path(None)?
            .with_context(|| "Unable to get the current block path while flushing")?;
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(current_block)?;

        while let Some(item) = tmp.pop_front() {
            let formatted_line = self.format_line_for_block(item.clone())?;
            if let Err(e) = writeln!(file, "{},{}", formatted_line.0, formatted_line.1) {
                retry_vec.push(item)
            }
        }

        Ok(())
    }

    fn format_line_for_block(&self, data: T) -> Result<(i64, String)> {
        let mut wtr = csv::Writer::from_writer(Vec::new());
        wtr.serialize(&data)?;
        let str = String::from_utf8(wtr.into_inner()?)?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as i64;
        Ok((now, str))
    }
}
