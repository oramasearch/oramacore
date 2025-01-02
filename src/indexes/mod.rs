pub mod bool;
pub mod number;
pub mod string;
pub mod vector;

// START FROM HERE:
// Task: `commit` action shouldn't block insertions & searches.
//
// The current implementation of `commit` is blocking because it accepts a `&mut self`,
// so there's a Mutext (or RwLock) somewhere.
// This is not good because it could block the access to the index for a long time.
// To achieve this, we need to:
// - Align number, vector to string implementations
//   This means we have a unique to invoke the `commit` method on all field indexes.
//   This allows also us to have a unique "view"/"way" to interact with the indexes during the commit phase.
// - The commit method should accept a `&self`.
//   This imply that we have to implement an internal locking mechanism.
// We have to avoid a lock an index for a too long period, so,
// the searches and insertion is not blocked for a long time.
//
// In general, we have:
// - committed index saved on disk
// - uncommitted index in memory
// *The hard part*: the in memory data is "live", so insertion is possible at any time.
//
// The committed index can be streamed from disk, so it's not a problem.
// Because we mixed the committed and uncommitted data, the uncommitted data
// is "locked" during the whole commit phase.
// This is not good because it could block searches and insertions.
//
// To avoid this, we can:
// - duplicate the uncommitted data (the data is smaller that the committed one)
// - insert to another uncommitted data
// - perform the searched on the two uncommitted data and the committed one
//
// So in general:
// - commit data
// - primary uncommitted data where searches and insertions are performed
// During the commit phase:
// - the searches are performed on the primary uncommitted data
// - create a second uncommitted data to store the new insertions
// - the searches are performed on primary uncommitted data, second uncommitted data and committed data
// - stream the primary uncommitted data with committed data and save it to disk
// - move the committed data pointer to the new data
//
