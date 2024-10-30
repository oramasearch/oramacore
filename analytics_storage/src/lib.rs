use anyhow::{anyhow, Context, Result};
use ms_converter::ms;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

pub type TimeSeriesData<T> = Vec<(i64, T)>;

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

pub struct AnalyticsStorageConfig {
    index_id: String,
    buffer_size: Option<usize>,
    granularity: Option<Granularity>,
    persistence_dir: Option<String>,
    offload_after: Option<usize>,
    offload_to: Option<OffloadTarget>,
}

struct AnalyticsStorage {
    index_id: String,
    docs_size: usize,
    buffer_size: usize,
    granularity: Granularity,
    persistence_dir: String,
    offload_after: usize,
    offload_to: OffloadTarget,
}

enum Version {
    V1_0,
}

struct VersionV1_0Schema {
    ID: String,
    DeploymentID: String,
    InstanceID: String,
    Timestamp: i64,
    RawSearchString: String,
    RawQuery: String,
    ResultsCount: usize,
    Referer: String,
    POP: String,
    Country: String,
    Continent: String,
    VisitorID: String,
}

struct TimeBlockMeta {
    doc_size: usize,
    version: Version,
    path: String,
    size: u64,
}

struct TimeBlock<T> {
    meta: TimeBlockMeta,
    data: TimeSeriesData<T>,
}

const ORAMA_ANALYTICS_DEFAULT_DIRNAME: &str = ".orama_analytics";
const DEFAULT_BUFFER_SIZE: usize = 100;
const DEFAULT_OFFLOAD_AFTER: usize = 100;

impl AnalyticsStorage {
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
        };

        let index_dir = storage.get_index_dir();
        let index_path = Path::new(&index_dir);

        if !index_path.is_dir() {
            fs::create_dir_all(index_path)
                .with_context(|| format!("Failed to create directory {}", index_dir))?;
        }

        Ok(storage)
    }

    fn get_block(&self, timestamp: i64) -> Result<TimeBlockMeta> {
        let index_dir = self.get_index_dir();

        let block_lifespan: (i64, i64) = match self.granularity {
            Granularity::Hour => {
                let hour = ms("1 hour")?;
                (timestamp - hour, timestamp + hour)
            }
            Granularity::Day => {
                let day = ms("1 day")?;
                (timestamp - day, timestamp + day)
            }
            Granularity::Week => {
                let week = ms("1 week")?;
                (timestamp - week, timestamp + week)
            }
            Granularity::Month => {
                let month = ms("30 days")?;
                (timestamp - month, timestamp + month)
            }
        };

        let block_path = fs::read_dir(index_dir.to_string())
            .with_context(|| format!("Failed to read directory {}", index_dir))?
            .map(|res| res.map(|e| e.path().display().to_string()))
            .find(|file| {
                if let Ok(file_path) = file {
                    if let Ok(time) = file_path.parse::<i64>() {
                        return time > block_lifespan.0 && time < block_lifespan.1;
                    }
                }
                false
            })
            .unwrap()?
            .to_string();

        Self::get_block_meta(&block_path)
    }

    fn get_index_dir(&self) -> String {
        format!("{}/{}", self.persistence_dir, self.index_id)
    }

    fn get_block_meta(path: &str) -> Result<TimeBlockMeta> {
        let file = fs::File::open(path).map_err(|_| anyhow!("Unable to open file {}", path))?;
        let file_size = file.metadata()?.len();

        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        if let Some(first_line) = lines.next() {
            let line =
                first_line.map_err(|_| anyhow!("Unable to read first line of file {}", path))?;
            let headers: Vec<&str> = line.split(',').collect();

            if headers.len() < 2 {
                return Err(anyhow!("Invalid header format in file {}", path));
            }

            let version = Self::parse_version(headers[0])?;
            let docs_size = headers[1]
                .parse::<usize>()
                .map_err(|_| anyhow!("Invalid doc_size in header of file {}", path))?;

            return Ok(TimeBlockMeta {
                version,
                doc_size: docs_size,
                size: file_size,
                path: path.to_string(),
            });
        }

        Err(anyhow!("Cannot find header for time block file {}", path))
    }

    fn parse_version(version: &str) -> Result<Version> {
        match version {
            "1.0" => Ok(Version::V1_0),
            _ => Err(anyhow!("Invalid time block version number: {}", version)),
        }
    }
}
