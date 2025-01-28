use std::{fmt::Debug, fs::File, path::PathBuf};

use anyhow::{Context, Result};
use fst::{automaton::StartsWith, Automaton, IntoStreamer, Map, MapBuilder, Streamer};
use memmap::Mmap;

use crate::file_utils::BufferedFile;

const FILE_NAME: &str = "index.fst";

pub struct FSTIndex {
    inner: Map<Mmap>,
}

impl Debug for FSTIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self
            .search_with_key("")
            .map(|(key, v)| (String::from_utf8_lossy(&key).to_string(), v))
            .collect::<Vec<_>>();
        f.debug_struct("FSTIndex").field("items", &s).finish()
    }
}

impl FSTIndex {
    pub fn from_iter<I, K, F>(
        iter: I,
        data_dir: PathBuf,
        mut f: F
    ) -> Result<Self>
    where
        I: Iterator<Item = (K, u64)>,
        K: AsRef<[u8]>,
        F: FnMut(&[u8], u64) -> ()
    {
        std::fs::create_dir_all(&data_dir)
            .context("Cannot create the base directory for the committed index")?;

        let path_to_commit = data_dir.join(FILE_NAME);

        let mut buffered_file = BufferedFile::create(path_to_commit.clone()).context("Cannot create file")?;
        let mut build = MapBuilder::new(&mut buffered_file)?;

        for (key, value) in iter {
            f(key.as_ref(), value);

            build
                .insert(key, value)
                .context("Cannot insert value to FST map")?;
        }

        build.finish().context("Cannot finish build of FST map")?;
        buffered_file.close().context("Cannot close buffered file")?;

        Self::load(data_dir)
    }

    pub fn load(data_dir: PathBuf) -> Result<Self> {
        let path_to_commit = data_dir.join(FILE_NAME);

        let file = File::open(path_to_commit)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let inner = Map::new(mmap)?;

        Ok(Self { inner })
    }

    pub fn search<'s, 'input>(&'s self, token: &'input str) -> FTSIter<'s, 'input>
    where
        'input: 's,
    {
        let automaton = fst::automaton::Str::new(token).starts_with();
        let stream: fst::map::Stream<'_, StartsWith<fst::automaton::Str<'_>>> =
            self.inner.search(automaton).into_stream();

        FTSIter {
            stream: Some(stream),
        }
    }

    pub fn search_with_key<'s, 'input>(&'s self, token: &'input str) -> FTSIterWithKey<'s, 'input>
    where
        'input: 's,
    {
        let automaton = fst::automaton::Str::new(token).starts_with();
        let stream: fst::map::Stream<'_, StartsWith<fst::automaton::Str<'_>>> =
            self.inner.search(automaton).into_stream();

        FTSIterWithKey {
            stream: Some(stream),
        }
    }
}

pub struct FTSIter<'stream, 'input> {
    stream: Option<fst::map::Stream<'stream, StartsWith<fst::automaton::Str<'input>>>>,
}
impl<'s, 'input> Iterator for FTSIter<'s, 'input> {
    // The Item allocate memory, but we could avoid it by using a reference
    // TODO: resolve lifetime issue with reference here
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let stream = match &mut self.stream {
            Some(stream) => stream,
            None => return None,
        };
        stream.next().map(|(_, value)| value)
    }
}


pub struct FTSIterWithKey<'stream, 'input> {
    stream: Option<fst::map::Stream<'stream, StartsWith<fst::automaton::Str<'input>>>>,
}
impl<'s, 'input> Iterator for FTSIterWithKey<'s, 'input> {
    // The Item allocate memory, but we could avoid it by using a reference
    // TODO: resolve lifetime issue with reference here
    type Item = (Vec<u8>, u64);

    fn next(&mut self) -> Option<Self::Item> {
        let stream = match &mut self.stream {
            Some(stream) => stream,
            None => return None,
        };
        stream.next()
            .map(|(key, value)| (key.to_vec(), value))
    }
}


#[cfg(test)]
mod tests {
    use crate::test_utils::generate_new_path;

    use super::*;

    #[test]
    fn test_fst_index() -> Result<()> {
        let data = vec![
            ("bar".as_bytes(), 3),
            ("far".as_bytes(), 2),
            ("foo".as_bytes(), 1),
        ];
        let data_dir = generate_new_path();
        let paged_index = FSTIndex::from_iter(data.into_iter(), data_dir.clone(), |_, _| {})?;
        test(&paged_index)?;

        let paged_index = FSTIndex::load(data_dir)?;
        test(&paged_index)?;

        fn test(paged_index: &FSTIndex) -> Result<()> {
            assert_eq!(
                paged_index.search("f").collect::<Vec<_>>(),
                vec![2, 1]
            );
            Ok(())
        }

        Ok(())
    }
}
