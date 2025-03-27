use std::{
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use tokio::io::AsyncWriteExt;
use tracing::trace;

pub async fn create_if_not_exists_async<P: AsRef<Path>>(p: P) -> Result<()> {
    let p: PathBuf = p.as_ref().to_path_buf();

    let output = tokio::fs::try_exists(&p).await;
    match output {
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Error while checking if the directory exists: {:?}",
                e
            ));
        }
        Ok(true) => {
            trace!("Directory exists. Skip creation.");
        }
        Ok(false) => {
            trace!("Directory does not exist. Creating it.");
            tokio::fs::create_dir_all(p)
                .await
                .context("Cannot create directory")?;
        }
    }

    Ok(())
}

pub fn create_if_not_exists<P: AsRef<Path>>(p: P) -> Result<()> {
    let p: PathBuf = p.as_ref().to_path_buf();

    match std::fs::exists(&p) {
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Error while checking if the directory exists: {:?}",
                e
            ));
        }
        Ok(true) => {
            trace!("Directory exists. Skip creation.");
        }
        Ok(false) => {
            trace!("Directory does not exist. Creating it.");
            std::fs::create_dir_all(p).context("Cannot create directory")?;
        }
    };

    Ok(())
}

pub async fn create_or_overwrite<T: serde::Serialize>(path: PathBuf, data: &T) -> Result<()> {
    let mut file = tokio::fs::File::create(&path)
        .await
        .with_context(|| format!("Cannot create file at {:?}", path))?;
    let v = serde_json::to_vec(data)
        .with_context(|| format!("Cannot write json data to {:?}", path))?;
    file.write_all(&v)
        .await
        .with_context(|| format!("Cannot write json data to {:?}", path))?;
    file.flush()
        .await
        .with_context(|| format!("Cannot flush file {:?}", path))?;
    file.sync_all()
        .await
        .with_context(|| format!("Cannot sync_all file {:?}", path))?;

    Ok(())
}

pub async fn read_file<T: serde::de::DeserializeOwned>(path: PathBuf) -> Result<T> {
    let vec = tokio::fs::read(&path)
        .await
        .with_context(|| format!("Cannot open file at {:?}", path))?;
    serde_json::from_slice(&vec)
        .with_context(|| format!("Cannot deserialize json data from {:?}", path))
}

pub struct BufferedFile;
impl BufferedFile {
    pub fn create_or_overwrite(path: PathBuf) -> Result<WriteBufferedFile> {
        let file = std::fs::File::create(&path)
            .with_context(|| format!("Cannot create file at {:?}", path))?;
        let buf = std::io::BufWriter::new(file);
        Ok(WriteBufferedFile {
            path,
            closed: false,
            buf,
        })
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<ReadBufferedFile> {
        Ok(ReadBufferedFile {
            path: path.as_ref().to_path_buf(),
        })
    }
}

pub struct ReadBufferedFile {
    path: PathBuf,
}

impl ReadBufferedFile {
    pub fn read_json_data<T: serde::de::DeserializeOwned>(self) -> Result<T> {
        let file = std::fs::File::open(&self.path)
            .with_context(|| format!("Cannot open file at {:?}", self.path))?;
        let reader = std::io::BufReader::new(file);
        let data = serde_json::from_reader(reader)
            .with_context(|| format!("Cannot read json data from {:?}", self.path))?;
        Ok(data)
    }

    pub fn read_bincode_data<T: serde::de::DeserializeOwned>(self) -> Result<T> {
        let file = std::fs::File::open(&self.path)
            .with_context(|| format!("Cannot open file at {:?}", self.path))?;
        let reader = std::io::BufReader::new(file);
        let data = bincode::deserialize_from(reader)
            .with_context(|| format!("Cannot read bincode data from {:?}", self.path))?;
        Ok(data)
    }
}

pub struct WriteBufferedFile {
    path: PathBuf,
    buf: std::io::BufWriter<std::fs::File>,
    closed: bool,
}
impl WriteBufferedFile {
    pub fn write_json_data<T: serde::Serialize>(mut self, data: &T) -> Result<()> {
        serde_json::to_writer(&mut self.buf, data)
            .with_context(|| format!("Cannot write json data to {:?}", self.path))?;
        self.close()
    }

    pub fn write_bincode_data<T: serde::Serialize>(mut self, data: &T) -> Result<()> {
        bincode::serialize_into(&mut self.buf, data)
            .with_context(|| format!("Cannot write bincode data to {:?}", self.path))?;
        self.close()
    }

    pub fn close(mut self) -> Result<()> {
        self.drop_all()
    }

    fn drop_all(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }

        self.closed = true;

        self.buf
            .flush()
            .with_context(|| format!("Cannot flush buffer {:?}", self.path))?;

        let mut inner = self.buf.get_ref();
        inner
            .flush()
            .with_context(|| format!("Cannot flush file {:?}", self.path))?;
        inner
            .sync_all()
            .with_context(|| format!("Cannot sync_all file {:?}", self.path))?;

        Ok(())
    }
}

// Proxy all std::io::Write methods to the inner buffer
impl std::io::Write for WriteBufferedFile {
    fn by_ref(&mut self) -> &mut Self {
        self.buf.by_ref();
        self
    }

    fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> std::io::Result<usize> {
        self.buf.write_vectored(bufs)
    }

    fn write_fmt(&mut self, fmt: std::fmt::Arguments<'_>) -> std::io::Result<()> {
        self.buf.write_fmt(fmt)
    }

    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.buf.write_all(buf)
    }

    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buf.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.buf.flush()
    }
}

impl Drop for WriteBufferedFile {
    fn drop(&mut self) {
        self.drop_all()
            .expect("WriteBufferedFile drop failed. Use `close` method to handle errors.");
    }
}
