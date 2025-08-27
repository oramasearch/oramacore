use std::{fmt::Debug, fs::File, path::PathBuf};

use anyhow::{Context, Result};
use fst::{
    automaton::{Levenshtein, StartsWith},
    Automaton, IntoStreamer, Map, MapBuilder, Streamer,
};
use memmap::Mmap;

use fs::BufferedFile;

pub struct FSTIndex {
    inner: Map<Mmap>,
    file_path: PathBuf,
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
    pub fn from_iter<I, K>(iter: I, file_path: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (K, u64)>,
        K: AsRef<[u8]>,
    {
        std::fs::create_dir_all(file_path.parent().expect("file path has no parent"))
            .context("Cannot create the base directory for the committed index")?;

        let mut buffered_file =
            BufferedFile::create_or_overwrite(file_path.clone()).context("Cannot create file")?;
        let mut build = MapBuilder::new(&mut buffered_file)?;

        for (key, value) in iter {
            build
                .insert(key, value)
                .context("Cannot insert value to FST map")?;
        }

        build.finish().context("Cannot finish build of FST map")?;
        buffered_file
            .close()
            .context("Cannot close buffered file")?;

        Self::load(file_path)
    }

    pub fn load(file_path: PathBuf) -> Result<Self> {
        let file = File::open(&file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let inner = Map::new(mmap)?;

        Ok(Self { inner, file_path })
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn file_path(&self) -> PathBuf {
        self.file_path.clone()
    }

    pub fn search<'s, 'input>(
        &'s self,
        token: &'input str,
        tolerance: Option<u8>,
    ) -> Result<Box<dyn Iterator<Item = (bool, u64)> + 's>>
    where
        'input: 's,
    {
        if let Some(tolerance) = tolerance {
            let automaton = fst::automaton::Levenshtein::new(token, u32::from(tolerance))
                .context("Cannot create Levenshtein automaton")?;
            let stream: fst::map::Stream<'_, Levenshtein> =
                self.inner.search(automaton).into_stream();

            Ok(Box::new(FTSLevenshteinIter {
                term: token.as_bytes(),
                stream: Some(stream),
            }))
        } else {
            let automaton = fst::automaton::Str::new(token).starts_with();
            let stream: fst::map::Stream<'_, StartsWith<fst::automaton::Str<'_>>> =
                self.inner.search(automaton).into_stream();

            Ok(Box::new(FTSIter {
                term: token.as_bytes(),
                stream: Some(stream),
            }))
        }
    }

    pub fn search_exact<'s, 'input>(&'s self, token: &'input str) -> Option<u64>
    where
        'input: 's,
    {
        self.inner.get(token)
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

    pub fn iter(&self) -> impl Iterator<Item = (Vec<u8>, u64)> + '_ {
        self.search_with_key("")
    }
}

pub struct FTSIter<'stream, 'input> {
    term: &'input [u8],
    stream: Option<fst::map::Stream<'stream, StartsWith<fst::automaton::Str<'input>>>>,
}
impl Iterator for FTSIter<'_, '_> {
    type Item = (bool, u64);

    fn next(&mut self) -> Option<Self::Item> {
        let stream = match &mut self.stream {
            Some(stream) => stream,
            None => return None,
        };
        stream.next().map(|(m, value)| (m == self.term, value))
    }
}

pub struct FTSLevenshteinIter<'stream, 'input> {
    term: &'input [u8],
    stream: Option<fst::map::Stream<'stream, fst::automaton::Levenshtein>>,
}
impl Iterator for FTSLevenshteinIter<'_, '_> {
    type Item = (bool, u64);

    fn next(&mut self) -> Option<Self::Item> {
        let stream = match &mut self.stream {
            Some(stream) => stream,
            None => return None,
        };
        stream.next().map(|(m, value)| (m == self.term, value))
    }
}

pub struct FTSIterWithKey<'stream, 'input> {
    stream: Option<fst::map::Stream<'stream, StartsWith<fst::automaton::Str<'input>>>>,
}
impl Iterator for FTSIterWithKey<'_, '_> {
    // The Item allocate memory, but we could avoid it by using a reference
    // TODO: resolve lifetime issue with reference here
    type Item = (Vec<u8>, u64);

    fn next(&mut self) -> Option<Self::Item> {
        let stream = match &mut self.stream {
            Some(stream) => stream,
            None => return None,
        };
        stream.next().map(|(key, value)| (key.to_vec(), value))
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::utils::generate_new_path;

    use super::*;

    #[test]
    fn test_fst_index() -> Result<()> {
        let data = vec![
            ("bar".as_bytes(), 3),
            ("far".as_bytes(), 2),
            ("foo".as_bytes(), 1),
        ];
        let data_dir = generate_new_path();
        let paged_index = FSTIndex::from_iter(data.into_iter(), data_dir.clone())?;
        test(&paged_index)?;

        let paged_index = FSTIndex::load(data_dir)?;
        test(&paged_index)?;

        fn test(paged_index: &FSTIndex) -> Result<()> {
            assert_eq!(
                paged_index.search("f", None).unwrap().collect::<Vec<_>>(),
                vec![(false, 2), (false, 1)]
            );
            Ok(())
        }

        Ok(())
    }

    #[test]
    fn test_fst_index_exact() -> Result<()> {
        let data = vec![
            ("bar".as_bytes(), 3),
            ("far".as_bytes(), 2),
            ("foo".as_bytes(), 1),
        ];
        let data_dir = generate_new_path();
        let paged_index = FSTIndex::from_iter(data.into_iter(), data_dir.clone())?;

        let output = paged_index.search_exact("f");
        assert_eq!(output, None);

        let output = paged_index.search_exact("foo");
        assert_eq!(output, Some(1));

        let output = paged_index.search_exact("foof");
        assert_eq!(output, None);

        let paged_index = FSTIndex::load(data_dir)?;

        let output = paged_index.search_exact("f");
        assert_eq!(output, None);

        let output = paged_index.search_exact("foo");
        assert_eq!(output, Some(1));

        let output = paged_index.search_exact("foof");
        assert_eq!(output, None);

        Ok(())
    }

    #[test]
    fn test_fst_index_tolerance() -> Result<()> {
        let data = vec![
            ("first".as_bytes(), 1),
            ("main".as_bytes(), 2),
            ("street".as_bytes(), 3),
        ];
        let data_dir = generate_new_path();
        let paged_index = FSTIndex::from_iter(data.into_iter(), data_dir.clone())?;

        // No tolerance
        let output: Vec<(bool, u64)> = paged_index.search("firt", Some(0))?.collect();
        assert_eq!(output, vec![]);
        // No tolerance
        let output: Vec<(bool, u64)> = paged_index.search("firt", None)?.collect();
        assert_eq!(output, vec![]);

        // With tolerance 1
        let output: Vec<_> = paged_index.search("firt", Some(1))?.collect();
        assert_eq!(output, vec![(false, 1)]);

        let output: Vec<_> = paged_index.search("firsta", Some(1))?.collect();
        assert_eq!(output, vec![(false, 1)]);

        Ok(())
    }

    #[test]
    fn test_exact_match() -> Result<()> {
        let data = vec![("serve".as_bytes(), 1), ("server".as_bytes(), 2)];
        let data_dir = generate_new_path();
        let paged_index = FSTIndex::from_iter(data.into_iter(), data_dir.clone())?;

        // No tolerance
        let output: Vec<(bool, u64)> = paged_index.search("serve", None)?.collect();
        assert_eq!(output, vec![(true, 1), (false, 2)]);

        Ok(())
    }
}
