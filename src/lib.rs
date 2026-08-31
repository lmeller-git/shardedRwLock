//! Sharded RwLock supporting generic scheduling strategies.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![deny(missing_docs)]
#![deny(clippy::missing_safety_doc, clippy::undocumented_unsafe_blocks)]
#![warn(unsafe_op_in_unsafe_fn)]

#[cfg(any(feature = "std", test))]
extern crate std;

#[allow(unused_extern_crates)]
#[cfg(any(feature = "alloc", test))]
extern crate alloc;

mod sync;

use core::{
    cell::UnsafeCell,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use crossbeam_utils::CachePadded;
use kasino::{
    Bandit,
    BanditHandle,
    Collection,
    InlineBandit,
    InlineStorage,
    Signature,
    WithCapacity,
    storage::StorageBackend,
    strategy::{Hooked, Strategy},
};
#[cfg(feature = "alloc")]
use kasino::{BoxedBandit, BoxedStorage};

/// The count of registered readers on a shard.
#[derive(Debug)]
pub struct ReaderShard<T>(CachePadded<AtomicUsize>, PhantomData<T>);

impl<T> Default for ReaderShard<T> {
    fn default() -> Self {
        Self(Default::default(), PhantomData)
    }
}

impl<T> WithCapacity<1> for ReaderShard<T> {
    fn with_capacity() -> Self {
        Self(AtomicUsize::new(0).into(), PhantomData)
    }
}

/// The state used for acquiring a lock
pub struct LockInput<'a, T> {
    writer: &'a AtomicBool,
    data: NonNull<UnsafeCell<T>>,
}

impl<'a, T> Copy for LockInput<'a, T> {}
impl<'a, T> Clone for LockInput<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A read-only access to the data.
pub struct ReaderGuard<'a, 'b, T> {
    shard: &'b ReaderShard<T>,
    ptr: NonNull<T>,
    _life: PhantomData<&'a ()>,
}

impl<'a, 'b, T> Deref for ReaderGuard<'a, 'b, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // # Safety:
        // we have acquired synchornized read access to the data and may thus deref it to a shared borow
        unsafe { self.ptr.as_ref() }
    }
}

impl<'a, 'b, T> Drop for ReaderGuard<'a, 'b, T> {
    fn drop(&mut self) {
        self.shard.0.fetch_sub(1, Ordering::Release);
    }
}

/// Input to Collection::offer
pub struct ReaderShardOffer<T>(PhantomData<T>);

impl<T> Signature for ReaderShardOffer<T> {
    type Error<'a, 'b>
        = ()
    where
        Self: 'b;
    type Input<'a> = LockInput<'a, T>;
    type Output<'a, 'b>
        = ReaderGuard<'a, 'b, T>
    where
        Self: 'b;
}

/// A reader-writer access to the data
#[derive(Debug)]
pub struct WriteGuard<'a, T> {
    b: &'a AtomicBool,
    ptr: NonNull<T>,
}

impl<'a, T> Deref for WriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // # Safety:
        // we have acquired synchornized mutable access to the data and may thus deref it to a shared borow
        unsafe { self.ptr.as_ref() }
    }
}

impl<'a, T> DerefMut for WriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // # Safety:
        // we have acquired synchornized mutable access to the data and may thus deref it to an exclusive borow
        unsafe { self.ptr.as_mut() }
    }
}

impl<'a, T> Drop for WriteGuard<'a, T> {
    fn drop(&mut self) {
        self.b.store(false, Ordering::Release);
    }
}

/// Input to Collection::poll
pub struct WritePoll<T>(PhantomData<T>);

impl<T> Signature for WritePoll<T> {
    type Error<'a, 'b>
        = usize
    where
        Self: 'b;
    type Input<'a> = LockInput<'a, T>;
    type Output<'a, 'b>
        = WriteGuard<'a, T>
    where
        Self: 'b;
}

impl<T> Collection for ReaderShard<T> {
    type OfferSignature = ReaderShardOffer<T>;
    type PollSignature = WritePoll<T>;

    fn offer<'b, 'a>(
        &'b self,
        item: <Self::OfferSignature as Signature>::Input<'a>,
    ) -> Result<
        <Self::OfferSignature as Signature>::Output<'a, 'b>,
        <Self::OfferSignature as Signature>::Error<'a, 'b>,
    > {
        let old_writer = item.writer.load(Ordering::Acquire);
        if old_writer {
            return Err(());
        }
        self.0.fetch_add(1, Ordering::Release);
        let writer_now = item.writer.load(Ordering::Acquire);
        if writer_now {
            self.0.fetch_sub(1, Ordering::Release);
            Err(())
        } else {
            Ok(ReaderGuard {
                shard: self,
                ptr: item.data.cast(),
                _life: PhantomData,
            })
        }
    }

    fn poll<'a, 'b>(
        &'b self,
        _input: <Self::PollSignature as Signature>::Input<'a>,
    ) -> Result<
        <Self::PollSignature as Signature>::Output<'a, 'b>,
        <Self::PollSignature as Signature>::Error<'a, 'b>,
    > {
        Err(self.0.load(Ordering::Acquire))
    }

    fn len(&self) -> usize {
        1
    }

    fn capacity(&self) -> usize {
        1
    }

    fn is_empty(&self) -> bool {
        self.0.load(Ordering::Acquire) == 0
    }
}

#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
struct RwLockStrategy<S>(S);

impl<T, S: Strategy<ReaderShard<T>>> Strategy<ReaderShard<T>> for RwLockStrategy<S> {
    type Gambler = S::Gambler;

    fn choose_offer_arm(
        &self,
        state: &impl StorageBackend<<Self::Gambler as Hooked>::Stake>,
        arm: &mut Self::Gambler,
    ) -> usize {
        self.0.choose_offer_arm(state, arm)
    }

    fn choose_poll_arm(
        &self,
        state: &impl StorageBackend<<Self::Gambler as Hooked>::Stake>,
        arm: &mut Self::Gambler,
    ) -> usize {
        self.0.choose_poll_arm(state, arm)
    }

    fn fork_gambler(&self, arm: &mut Self::Gambler) -> Self::Gambler {
        self.0.fork_gambler(arm)
    }

    fn create_gambler(&self) -> Self::Gambler {
        self.0.create_gambler()
    }

    fn collect<'b, 'c>(
        &self,
        _state: &impl StorageBackend<<Self::Gambler as Hooked>::Stake>,
        sub_collections: &'c impl StorageBackend<ReaderShard<T>>,
        input: <<ReaderShard<T> as Collection>::PollSignature as Signature>::Input<'b>,
    ) -> Option<(
        <<ReaderShard<T> as Collection>::PollSignature as Signature>::Output<'b, 'c>,
        usize,
    )>
    where
        ReaderShard<T>: 'c,
    {
        fn no_reader<'b, 'c, T>(
            sub_collections: &'c impl StorageBackend<ReaderShard<T>>,
            input: LockInput<'b, T>,
        ) -> bool {
            sub_collections
                .iter()
                .all(|item| matches!(item.poll(input), Err(0)))
        }

        if !no_reader(sub_collections, input) {
            return None;
        }

        if input.writer.swap(true, Ordering::AcqRel) {
            return None;
        }

        if !no_reader(sub_collections, input) {
            input.writer.store(false, Ordering::Release);
            return None;
        }

        Some((
            WriteGuard {
                b: input.writer,
                ptr: input.data.cast(),
            },
            0,
        ))
    }
}

///  A sharded RwLock
pub struct ShardedRwLock<
    T,
    B: StorageBackend<ReaderShard<T>>,
    C: StorageBackend<<S::Gambler as Hooked>::Stake>,
    S: Strategy<ReaderShard<T>>,
> {
    shards: Bandit<ReaderShard<T>, RwLockStrategy<S>, B, C, 1>,
    writer: AtomicBool,
    item: UnsafeCell<T>,
}

impl<T, B, C, S> ShardedRwLock<T, B, C, S>
where
    B: StorageBackend<ReaderShard<T>>,
    C: StorageBackend<<S::Gambler as Hooked>::Stake>,
    S: Strategy<ReaderShard<T>>,
{
    /// Acquires a new ShardedRwLockHandle to this object
    pub fn new_root(&self) -> ShardedRwLockHandle<'_, T, B, C, S> {
        ShardedRwLockHandle {
            shards_handle: self.shards.buy_in(),
            parent: self,
        }
    }
}

// # Safety:
// This is safe if all types are sync
unsafe impl<T: Sync, B: Sync, C: Sync, S: Sync> Sync for ShardedRwLock<T, B, C, S>
where
    B: StorageBackend<ReaderShard<T>>,
    C: StorageBackend<<S::Gambler as Hooked>::Stake>,
    S: Strategy<ReaderShard<T>>,
{
}

// # Safety:
// This is safe if all types are send
unsafe impl<T: Send, B: Send, C: Send, S: Send> Send for ShardedRwLock<T, B, C, S>
where
    B: StorageBackend<ReaderShard<T>>,
    C: StorageBackend<<S::Gambler as Hooked>::Stake>,
    S: Strategy<ReaderShard<T>>,
{
}

/// A handle to a ShardedRwLock
pub struct ShardedRwLockHandle<'a, T, B, C, S: Strategy<ReaderShard<T>>>
where
    B: StorageBackend<ReaderShard<T>>,
    C: StorageBackend<<S::Gambler as Hooked>::Stake>,
{
    shards_handle: BanditHandle<'a, ReaderShard<T>, RwLockStrategy<S>, B, C, 1>,
    parent: &'a ShardedRwLock<T, B, C, S>,
}

impl<'a, T, B, C, S: Strategy<ReaderShard<T>>> ShardedRwLockHandle<'a, T, B, C, S>
where
    B: StorageBackend<ReaderShard<T>>,
    C: StorageBackend<<S::Gambler as Hooked>::Stake>,
{
    /// Attempts to acquire read only access to the data
    pub fn read(&mut self) -> Option<ReaderGuard<'a, '_, T>> {
        self.shards_handle
            .offer(LockInput {
                writer: &self.parent.writer,
                data: NonNull::from(&self.parent.item),
            })
            .ok()
    }

    /// Attempts to acquire reader-writer access to the data
    pub fn write(&mut self) -> Option<WriteGuard<'a, T>> {
        self.shards_handle
            .poll(LockInput {
                writer: &self.parent.writer,
                data: NonNull::from(&self.parent.item),
            })
            .ok()
    }

    /// Creates a new ShardedRwLockHandle from this handle
    pub fn fork(&mut self) -> Self {
        Self {
            shards_handle: self.shards_handle.fork(),
            parent: self.parent,
        }
    }
}

/// A ShardedRwLock that is dynamically stored
#[expect(type_alias_bounds)]
#[cfg(feature = "alloc")]
pub type BoxedShardedRwLock<T, S: Strategy<ReaderShard<T>>> =
    ShardedRwLock<T, BoxedStorage<ReaderShard<T>>, BoxedStorage<<S::Gambler as Hooked>::Stake>, S>;

#[cfg(feature = "alloc")]
impl<T, S> BoxedShardedRwLock<T, S>
where
    S: Strategy<ReaderShard<T>> + Default,
{
    /// Constructs a new BoxedShardedRwLock with shard count chsard_count
    pub fn new(shard_count: usize, item: T) -> Self {
        Self {
            shards: BoxedBandit::new(shard_count),
            writer: AtomicBool::new(false),
            item: UnsafeCell::new(item),
        }
    }
}

/// A ShardedRwLock stored inline
#[expect(type_alias_bounds)]
pub type InlineShardedRwLock<T, S: Strategy<ReaderShard<T>>, const N: usize> = ShardedRwLock<
    T,
    InlineStorage<ReaderShard<T>, N>,
    InlineStorage<<S::Gambler as Hooked>::Stake, N>,
    S,
>;

impl<T, S, const N: usize> InlineShardedRwLock<T, S, N>
where
    S: Strategy<ReaderShard<T>> + Default,
{
    /// Constructs a new InlineShardedRwLock
    pub fn new(item: T) -> Self {
        Self {
            shards: InlineBandit::new(),
            writer: AtomicBool::new(false),
            item: UnsafeCell::new(item),
        }
    }
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Barrier},
        thread,
    };

    use super::*;

    // --- Minimal Mock Strategyr for Testing ---
    #[derive(Default, Debug, Clone, Copy)]
    struct TestStrategy;

    pub(crate) struct TestArm;
    impl Hooked for TestArm {
        type Stake = ();
    }

    impl<T> Strategy<ReaderShard<T>> for TestStrategy {
        type Gambler = TestArm;

        fn choose_offer_arm(
            &self,
            _state: &impl StorageBackend<<Self::Gambler as Hooked>::Stake>,
            _arm: &mut Self::Gambler,
        ) -> usize {
            0
        }

        fn choose_poll_arm(
            &self,
            _choose_to: &impl StorageBackend<<Self::Gambler as Hooked>::Stake>,
            _arm: &mut Self::Gambler,
        ) -> usize {
            0
        }

        fn fork_gambler(&self, _arm: &mut Self::Gambler) -> Self::Gambler {
            TestArm
        }

        fn create_gambler(&self) -> Self::Gambler {
            TestArm
        }

        fn collect<'b, 'c>(
            &self,
            _state: &impl StorageBackend<<Self::Gambler as Hooked>::Stake>,
            _sub_collections: &'c impl StorageBackend<ReaderShard<T>>,
            _input: <<ReaderShard<T> as Collection>::PollSignature as Signature>::Input<'b>,
        ) -> Option<(
            <<ReaderShard<T> as Collection>::PollSignature as Signature>::Output<'b, 'c>,
            usize,
        )>
        where
            ReaderShard<T>: 'c,
        {
            unreachable!()
        }
    }

    impl<T> BoxedShardedRwLock<T, TestStrategy> {
        pub(crate) fn test_new(val: T) -> Self {
            Self {
                shards: BoxedBandit::new(8),
                writer: AtomicBool::new(false),
                item: UnsafeCell::new(val),
            }
        }
    }

    // --- Tests ---

    #[test]
    fn test_basic_read_and_write() {
        let lock = ShardedRwLock::test_new(42);
        let mut handle = lock.new_root();

        // Read initial value
        {
            let guard = handle.read().expect("Failed to acquire read lock");
            assert_eq!(*guard, 42);
        }

        // Mutate value
        {
            let mut guard = handle.write().expect("Failed to acquire write lock");
            *guard = 100;
        }

        // Verify mutation
        {
            let guard = handle.read().expect("Failed to acquire read lock");
            assert_eq!(*guard, 100);
        }
    }

    #[test]
    fn test_concurrent_readers() {
        let lock = ShardedRwLock::test_new("shared data");
        let mut h1 = lock.new_root();
        let mut h2 = h1.fork();

        let r1 = h1.read().expect("First reader failed");
        let r2 = h2.read().expect("Second reader failed concurrently");

        assert_eq!(*r1, "shared data");
        assert_eq!(*r2, "shared data");
    }

    #[test]
    fn test_write_exclusion_with_active_readers() {
        let lock = ShardedRwLock::test_new(10);
        let mut h1 = lock.new_root();
        let mut h2 = h1.fork();

        let _reader = h1.read().expect("Reader failed");

        // Write attempt must fail while reader exists
        assert!(h2.write().is_none());
    }

    #[test]
    fn test_exclusion_during_active_writer() {
        let lock = ShardedRwLock::test_new(10);
        let mut h1 = lock.new_root();
        let mut h2 = h1.fork();

        let _writer = h1.write().expect("Writer failed");

        // Both read and write must fail while writer exists
        assert!(h2.read().is_none());
        assert!(h2.write().is_none());
    }

    #[test]
    fn test_guard_drop_releases_lock() {
        let lock = ShardedRwLock::test_new(0);
        let mut h1 = lock.new_root();
        let mut h2 = h1.fork();

        let writer = h1.write().unwrap();
        drop(writer);

        // Lock freed: reader should succeed
        let reader = h2.read();
        assert!(reader.is_some());
        drop(reader);

        // Lock freed again: write should succeed
        assert!(h1.write().is_some());
    }

    #[test]
    fn test_multithreaded_contention() {
        let lock = Arc::new(ShardedRwLock::test_new(0));
        let threads = 8;
        let iterations = 1000;
        let barrier = Arc::new(Barrier::new(threads));

        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let lock = Arc::clone(&lock);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    let mut handle = lock.new_root();
                    barrier.wait();

                    for _ in 0..iterations {
                        // Spin until write acquired
                        loop {
                            if let Some(mut guard) = handle.write() {
                                *guard += 1;
                                break;
                            }
                            thread::yield_now();
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let mut root = lock.new_root();
        let final_guard = root.read().unwrap();
        assert_eq!(*final_guard, threads * iterations);
    }
}
