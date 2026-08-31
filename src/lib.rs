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
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
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

use crate::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    cell::UnsafeCell,
};

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

use sealed::*;
mod sealed {
    use super::*;

    /// The state used for acquiring a lock
    #[expect(unnameable_types)]
    pub struct LockInput<'a, T> {
        pub(crate) writer: &'a AtomicBool,
        pub(crate) data: NonNull<UnsafeCell<T>>,
    }

    impl<'a, T> Copy for LockInput<'a, T> {}
    impl<'a, T> Clone for LockInput<'a, T> {
        fn clone(&self) -> Self {
            *self
        }
    }

    /// Input to Collection::offer
    #[expect(unnameable_types)]
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

    /// Input to Collection::poll
    #[expect(unnameable_types)]
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
    pub(crate) struct RwLockStrategy<S>(S);

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
#[cfg(feature = "alloc")]
#[expect(type_alias_bounds)]
pub type BoxedShardedRwLock<T, S: Strategy<ReaderShard<T>>> =
    ShardedRwLock<T, BoxedStorage<ReaderShard<T>>, BoxedStorage<<S::Gambler as Hooked>::Stake>, S>;

/// A handle to a BoxedShardedRwLock
#[cfg(feature = "alloc")]
#[expect(type_alias_bounds)]
pub type BoxedShardedRwLockHandle<'a, T, S: Strategy<ReaderShard<T>>> = ShardedRwLockHandle<
    'a,
    T,
    BoxedStorage<ReaderShard<T>>,
    BoxedStorage<<S::Gambler as Hooked>::Stake>,
    S,
>;

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

/// A handle to an InlineShardedRwLock
#[expect(type_alias_bounds)]
pub type InlineShardedRwLockHandle<'a, T, S: Strategy<ReaderShard<T>>, const N: usize> =
    ShardedRwLockHandle<
        'a,
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

#[cfg(test)]
mod tests {
    use super::*;

    trait ForkeableRwLockImpl<'a, T> {
        fn read(&mut self) -> Option<ReaderGuard<'a, '_, T>>;
        fn write(&mut self) -> Option<WriteGuard<'a, T>>;
        fn fork(&mut self) -> Self;
    }

    impl<'a, T, S: Strategy<ReaderShard<T>>, const N: usize> ForkeableRwLockImpl<'a, T>
        for InlineShardedRwLockHandle<'a, T, S, N>
    {
        fn read(&mut self) -> Option<ReaderGuard<'a, '_, T>> {
            self.read()
        }

        fn write(&mut self) -> Option<WriteGuard<'a, T>> {
            self.write()
        }

        fn fork(&mut self) -> Self {
            self.fork()
        }
    }

    fn smoke<'a, L>(lock: L)
    where
        L: ForkeableRwLockImpl<'a, i32>,
    {
        todo!()
    }

    fn many_reader<'a, L>(lock: L)
    where
        L: ForkeableRwLockImpl<'a, i32>,
    {
        todo!()
    }

    fn send_sync<L>(_lock: L)
    where
        L: Send + Sync,
    {
    }

    fn concurrent_read<'a, L>(lock: L)
    where
        L: ForkeableRwLockImpl<'a, i32>,
    {
        todo!()
    }

    fn concurrent_write<'a, L>(lock: L)
    where
        L: ForkeableRwLockImpl<'a, i32>,
    {
        todo!()
    }

    fn concurrent_rw<'a, L>(lock: L)
    where
        L: ForkeableRwLockImpl<'a, i32>,
    {
        todo!()
    }

    #[cfg(all(not(loom), not(shuttle)))]
    mod core {
        use kasino::strategy::{RandomAccess, RoundRobin};

        use super::*;
        use crate::tests::smoke;

        #[test]
        fn send_sync_inline() {
            send_sync(InlineShardedRwLock::<_, RoundRobin, 1>::new(0).new_root());
        }

        #[cfg(feature = "alloc")]
        fn send_sync_boxed() {
            send_sync(BoxedShardedRwLock::<_, RoundRobin>::new(1, 0).new_root());
        }

        #[test]
        fn smoke_impl() {
            smoke(InlineShardedRwLock::<_, RoundRobin, 10>::new(0).new_root());
        }

        #[test]
        fn many_reader_impl() {
            many_reader(InlineShardedRwLock::<_, RoundRobin, 10>::new(0).new_root());
        }

        #[test]
        fn drops_impl() {
            struct Drops<'a> {
                c: &'a AtomicUsize,
            }

            impl<'a> Drop for Drops<'a> {
                fn drop(&mut self) {
                    self.c.fetch_add(1, Ordering::Relaxed);
                }
            }

            let counter = AtomicUsize::new(0);

            drop(InlineShardedRwLock::<_, RoundRobin, 10>::new(Drops {
                c: &counter,
            }));

            assert_eq!(counter.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn concurrent_read_impl() {
            concurrent_read(InlineShardedRwLock::<_, RandomAccess, 10>::new(0).new_root());
        }

        #[test]
        fn concurrent_write_impl() {
            concurrent_write(InlineShardedRwLock::<_, RandomAccess, 10>::new(0).new_root());
        }

        #[test]
        fn concurrent_rw_impl() {
            concurrent_rw(InlineShardedRwLock::<_, RandomAccess, 10>::new(0).new_root());
        }
    }

    #[cfg(shuttle)]
    mod shuttle {
        use super::*;

        const ITER: usize = 100;
        const DEPTH: usize = 4;

        #[test]
        fn concurrent_read_impl() {
            shuttle::check_pct(
                || concurrent_read(InlineShardedRwLock::<_, RandomAccess, 10>::new(0).new_root()),
                ITER,
                DEPTH,
            )
        }

        #[test]
        fn concurrent_write_impl() {
            shuttle::check_pct(
                || concurrent_write(InlineShardedRwLock::<_, RandomAccess, 10>::new(0).new_root()),
                ITER,
                DEPTH,
            )
        }

        #[test]
        fn concurrent_rw_impl() {
            shuttle::check_pct(
                || concurrent_rw(InlineShardedRwLock::<_, RandomAccess, 10>::new(0).new_root()),
                ITER,
                DEPTH,
            )
        }
    }

    #[cfg(loom)]
    mod loom {
        use super::*;

        #[test]
        fn concurrent_read_impl() {
            loom::model(|| {
                concurrent_read(InlineShardedRwLock::<_, RandomAccess, 10>::new(0).new_root())
            })
        }

        #[test]
        fn concurrent_write_impl() {
            loom::model(|| {
                concurrent_write(InlineShardedRwLock::<_, RandomAccess, 10>::new(0).new_root())
            })
        }

        #[test]
        fn concurrent_rw_impl() {
            loom::model(|| {
                concurrent_rw(InlineShardedRwLock::<_, RandomAccess, 10>::new(0).new_root())
            })
        }
    }
}
