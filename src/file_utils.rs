use std::{io::Write, path::PathBuf};

use anyhow::{Context, Result};


pub struct BufferedFile {
    path: PathBuf,
    buf: std::io::BufWriter<std::fs::File>,
    closed: bool,
}

impl BufferedFile {
    pub fn create(path: PathBuf) -> Result<BufferedFile> {
        let file = std::fs::File::create_new(&path)
            .with_context(|| format!("Cannot create file at {:?}", path))?;
        let buf = std::io::BufWriter::new(file);
        Ok(BufferedFile {
            path,
            closed: false,
            buf
        })
    }

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

        self.buf.flush()
            .with_context(|| format!("Cannot flush buffer {:?}", self.path))?;

        let mut inner = self.buf.get_ref();
        inner.flush()
            .with_context(|| format!("Cannot flush file {:?}", self.path))?;
        inner.sync_all()
            .with_context(|| format!("Cannot sync_all file {:?}", self.path))?;

        Ok(())
    }
}

// Proxy all std::io::Write methods to the inner buffer
impl std::io::Write for BufferedFile {
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

impl Drop for BufferedFile {
    fn drop(&mut self) {
        self.drop_all().expect("BufferedFile drop failed. Use `close` method to handle errors.");
    }
}
