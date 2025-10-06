use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
    sync::{
        PoisonError, RwLock as StdRwLock, RwLockReadGuard as StdRwLockReadGuard,
        RwLockWriteGuard as StdRwLockWriteGuard,
    },
};

use tokio::sync::{
    Mutex as TokioMutex, MutexGuard, RwLock as TokioRwLock,
    RwLockReadGuard as TokioRwLockReadGuard, RwLockWriteGuard as TokioRwLockWriteGuard,
};

use crate::metrics::{
    histogram::TimeHistogramImpl,
    locks::{LOCKED_FOR_TIME, LOCKING_TIME},
    LockNameLabels, LockType,
};

pub struct OramaAsyncLock<T> {
    name: &'static str,
    inner: TokioRwLock<T>,
}

impl<T> OramaAsyncLock<T> {
    pub fn new(name: &'static str, data: T) -> Self {
        Self {
            name,
            inner: TokioRwLock::new(data),
        }
    }

    pub async fn read(&self, reason: &'static str) -> OramaAsyncLockReadGuard<'_, T> {
        let m = LOCKING_TIME.create(LockNameLabels {
            name: self.name,
            reason,
            lock_type: LockType::Read,
        });
        let guard = self.inner.read().await;
        drop(m);

        let m = LOCKED_FOR_TIME.create(LockNameLabels {
            name: self.name,
            reason,
            lock_type: LockType::Read,
        });

        OramaAsyncLockReadGuard { guard, m: Some(m) }
    }

    pub async fn write(&self, reason: &'static str) -> OramaAsyncLockWriteGuard<'_, T> {
        let m = LOCKING_TIME.create(LockNameLabels {
            name: self.name,
            reason,
            lock_type: LockType::Write,
        });
        let guard = self.inner.write().await;
        drop(m);

        let m = LOCKED_FOR_TIME.create(LockNameLabels {
            name: self.name,
            reason,
            lock_type: LockType::Write,
        });

        OramaAsyncLockWriteGuard { guard, m: Some(m) }
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.inner.get_mut()
    }
}

impl<T: Debug> Debug for OramaAsyncLock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OramaAsyncLock")
            .field("name", &self.name)
            .field("inner", &self.inner)
            .finish()
    }
}

pub struct OramaAsyncLockReadGuard<'lock, T> {
    guard: TokioRwLockReadGuard<'lock, T>,
    m: Option<TimeHistogramImpl>,
}
impl<'lock, T> Deref for OramaAsyncLockReadGuard<'lock, T> {
    type Target = TokioRwLockReadGuard<'lock, T>;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<'lock, T> Drop for OramaAsyncLockReadGuard<'lock, T> {
    fn drop(&mut self) {
        if let Some(m) = self.m.take() {
            drop(m);
        }
    }
}
pub struct OramaAsyncLockWriteGuard<'lock, T> {
    guard: TokioRwLockWriteGuard<'lock, T>,
    m: Option<TimeHistogramImpl>,
}

impl<'lock, T> Deref for OramaAsyncLockWriteGuard<'lock, T> {
    type Target = TokioRwLockWriteGuard<'lock, T>;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<'lock, T> DerefMut for OramaAsyncLockWriteGuard<'lock, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}
impl<'lock, T> Drop for OramaAsyncLockWriteGuard<'lock, T> {
    fn drop(&mut self) {
        if let Some(m) = self.m.take() {
            drop(m);
        }
    }
}

pub struct OramaAsyncMutex<T> {
    name: &'static str,
    inner: TokioMutex<T>,
}
impl<T> OramaAsyncMutex<T> {
    pub fn new(name: &'static str, data: T) -> Self {
        Self {
            name,
            inner: TokioMutex::new(data),
        }
    }

    pub async fn lock(&self, reason: &'static str) -> OramaAsyncMutexGuard<'_, T> {
        let m = LOCKING_TIME.create(LockNameLabels {
            name: self.name,
            reason,
            lock_type: LockType::Mutex,
        });
        let guard = self.inner.lock().await;
        drop(m);

        let m = LOCKED_FOR_TIME.create(LockNameLabels {
            name: self.name,
            reason,
            lock_type: LockType::Mutex,
        });

        OramaAsyncMutexGuard { guard, m: Some(m) }
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.inner.get_mut()
    }
}

pub struct OramaAsyncMutexGuard<'lock, T> {
    guard: MutexGuard<'lock, T>,
    m: Option<TimeHistogramImpl>,
}
impl<'lock, T> Deref for OramaAsyncMutexGuard<'lock, T> {
    type Target = MutexGuard<'lock, T>;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}
impl<'lock, T> DerefMut for OramaAsyncMutexGuard<'lock, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}
impl<'lock, T> Drop for OramaAsyncMutexGuard<'lock, T> {
    fn drop(&mut self) {
        if let Some(m) = self.m.take() {
            drop(m);
        }
    }
}
pub struct OramaSyncLock<T> {
    name: &'static str,
    inner: StdRwLock<T>,
}

impl<T> OramaSyncLock<T> {
    pub fn new(name: &'static str, data: T) -> Self {
        Self {
            name,
            inner: StdRwLock::new(data),
        }
    }

    pub fn read(&self, reason: &'static str) -> LockReadResult<'_, T> {
        let m = LOCKING_TIME.create(LockNameLabels {
            name: self.name,
            reason,
            lock_type: LockType::Read,
        });
        let guard = self.inner.read();
        drop(m);

        match guard {
            Ok(g) => {
                let m = LOCKED_FOR_TIME.create(LockNameLabels {
                    name: self.name,
                    reason,
                    lock_type: LockType::Read,
                });
                Ok(OramaSyncLockReadGuard {
                    guard: g,
                    m: Some(m),
                })
            }
            Err(e) => Err(e),
        }
    }

    pub fn write(&self, reason: &'static str) -> LockWriteResult<'_, T> {
        let m = LOCKING_TIME.create(LockNameLabels {
            name: self.name,
            reason,
            lock_type: LockType::Write,
        });
        let guard = self.inner.write();
        drop(m);

        match guard {
            Ok(g) => {
                let m = LOCKED_FOR_TIME.create(LockNameLabels {
                    name: self.name,
                    reason,
                    lock_type: LockType::Write,
                });
                Ok(OramaSyncLockWriteGuard {
                    guard: g,
                    m: Some(m),
                })
            }
            Err(e) => Err(e),
        }
    }
}

impl<T: Debug> Debug for OramaSyncLock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OramaSyncLock")
            .field("name", &self.name)
            .field("inner", &self.inner)
            .finish()
    }
}

pub struct OramaSyncLockReadGuard<'lock, T> {
    guard: StdRwLockReadGuard<'lock, T>,
    m: Option<TimeHistogramImpl>,
}
impl<'lock, T> Deref for OramaSyncLockReadGuard<'lock, T> {
    type Target = StdRwLockReadGuard<'lock, T>;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}
impl<'lock, T> Drop for OramaSyncLockReadGuard<'lock, T> {
    fn drop(&mut self) {
        if let Some(m) = self.m.take() {
            drop(m);
        }
    }
}

pub struct OramaSyncLockWriteGuard<'lock, T> {
    guard: StdRwLockWriteGuard<'lock, T>,
    m: Option<TimeHistogramImpl>,
}
impl<'lock, T> Deref for OramaSyncLockWriteGuard<'lock, T> {
    type Target = StdRwLockWriteGuard<'lock, T>;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}
impl<'lock, T> DerefMut for OramaSyncLockWriteGuard<'lock, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}
impl<'lock, T> Drop for OramaSyncLockWriteGuard<'lock, T> {
    fn drop(&mut self) {
        if let Some(m) = self.m.take() {
            drop(m);
        }
    }
}

pub type LockReadResult<'lock, T> =
    Result<OramaSyncLockReadGuard<'lock, T>, PoisonError<StdRwLockReadGuard<'lock, T>>>;
pub type LockWriteResult<'lock, T> =
    Result<OramaSyncLockWriteGuard<'lock, T>, PoisonError<StdRwLockWriteGuard<'lock, T>>>;
