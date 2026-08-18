use std::{
    cell::UnsafeCell,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use lope::{
    BoxedArm, BoxedLope, Collection, IODescription, NewSized,
    schedule::{Hooked, Schedule},
    storage::StorageBackend,
};

#[derive(Debug, Default)]
pub struct ReaderShard<T>(AtomicUsize, PhantomData<T>);

impl<T> NewSized<1> for ReaderShard<T> {
    fn with_capacity() -> Self {
        Self(AtomicUsize::new(0), PhantomData)
    }
}

pub struct LockInput<'a, T> {
    pub writer: &'a AtomicBool,
    pub data: NonNull<UnsafeCell<T>>,
    pub _marker: PhantomData<&'a ()>,
}

impl<'a, T> Copy for LockInput<'a, T> {}
impl<'a, T> Clone for LockInput<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

pub struct ReaderGuard<'a, 'b, T> {
    shard: &'b ReaderShard<T>,
    ptr: NonNull<T>,
    _life: PhantomData<&'a ()>,
}

impl<'a, 'b, T> Deref for ReaderGuard<'a, 'b, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<'a, 'b, T> Drop for ReaderGuard<'a, 'b, T> {
    fn drop(&mut self) {
        self.shard.0.fetch_sub(1, Ordering::Release);
    }
}

pub struct ReaderShardOffer<T>(PhantomData<T>);

impl<T> IODescription for ReaderShardOffer<T> {
    type Input<'a> = LockInput<'a, T>;
    type Output<'a, 'b>
        = ReaderGuard<'a, 'b, T>
    where
        Self: 'b;
    type Error<'a, 'b>
        = ()
    where
        Self: 'b;
}

#[derive(Debug)]
pub struct WriteGuard<'a, T> {
    b: &'a AtomicBool,
    ptr: NonNull<T>,
    _life: PhantomData<&'a ()>,
}

impl<'a, T> Deref for WriteGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<'a, T> DerefMut for WriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut() }
    }
}

impl<'a, T> Drop for WriteGuard<'a, T> {
    fn drop(&mut self) {
        self.b.store(false, Ordering::Release);
    }
}

pub struct WritePoll<T>(PhantomData<T>);

impl<T> IODescription for WritePoll<T> {
    type Input<'a> = LockInput<'a, T>;
    type Output<'a, 'b>
        = WriteGuard<'a, T>
    where
        Self: 'b;
    type Error<'a, 'b>
        = usize
    where
        Self: 'b;
}

impl<T> Collection for ReaderShard<T> {
    type OfferIO = ReaderShardOffer<T>;
    type PollIO = WritePoll<T>;

    fn offer<'b, 'a>(
        &'b self,
        item: <Self::OfferIO as IODescription>::Input<'a>,
    ) -> Result<
        <Self::OfferIO as IODescription>::Output<'a, 'b>,
        <Self::OfferIO as IODescription>::Error<'a, 'b>,
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
        _input: <Self::PollIO as IODescription>::Input<'a>,
    ) -> Result<
        <Self::PollIO as IODescription>::Output<'a, 'b>,
        <Self::PollIO as IODescription>::Error<'a, 'b>,
    > {
        Err(self.0.load(Ordering::Acquire))
    }

    fn len(&self) -> usize {
        1
    }

    fn cap(&self) -> usize {
        1
    }

    fn is_empty(&self) -> bool {
        self.0.load(Ordering::Acquire) == 0
    }
}

#[allow(unreachable_pub)]
pub trait View<'a, T> {
    fn project(&'a self) -> &'a T;
}

impl<'a, K, U, T> View<'a, T> for K
where
    K: Deref<Target = U>,
    U: View<'a, T> + 'a,
{
    fn project(&'a self) -> &'a T {
        U::project(self)
    }
}

#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
struct RWLockScheduler<S>(S);

impl<T, S: Schedule<ReaderShard<T>>> Schedule<ReaderShard<T>> for RWLockScheduler<S> {
    type Arm = S::Arm;

    fn choose_offer_shard(
        &self,
        state: &impl StorageBackend<<Self::Arm as Hooked>::State>,
        arm: &mut Self::Arm,
    ) -> usize {
        self.0.choose_offer_shard(state, arm)
    }

    fn choose_poll_shard(
        &self,
        state: &impl StorageBackend<<Self::Arm as Hooked>::State>,
        arm: &mut Self::Arm,
    ) -> usize {
        self.0.choose_poll_shard(state, arm)
    }

    fn fork_arm(&self, arm: &mut Self::Arm) -> Self::Arm {
        self.0.fork_arm(arm)
    }

    fn create_arm(&self) -> Self::Arm {
        self.0.create_arm()
    }

    fn collect<'b, 'c>(
        &self,
        _state: &impl StorageBackend<<Self::Arm as Hooked>::State>,
        sub_collections: &'c impl StorageBackend<ReaderShard<T>>,
        input: <<ReaderShard<T> as Collection>::PollIO as IODescription>::Input<'b>,
    ) -> Option<(
        <<ReaderShard<T> as Collection>::PollIO as IODescription>::Output<'b, 'c>,
        usize,
    )>
    where
        ReaderShard<T>: 'c,
    {
        for item in sub_collections.iter() {
            let Err(c) = item.poll(input) else {
                unreachable!();
            };
            if c != 0 {
                return None;
            }
        }

        if input.writer.swap(true, Ordering::AcqRel) {
            return None;
        }

        for item in sub_collections.iter() {
            let Err(c) = item.poll(input) else {
                unreachable!();
            };
            if c != 0 {
                input.writer.store(false, Ordering::Release);
                return None;
            }
        }

        Some((
            WriteGuard {
                b: input.writer,
                ptr: input.data.cast(),
                _life: PhantomData,
            },
            0,
        ))
    }
}

pub struct ShardedRwLock<T, S: Schedule<ReaderShard<T>>> {
    shards: BoxedLope<ReaderShard<T>, RWLockScheduler<S>, 1>,
    writer: AtomicBool,
    item: UnsafeCell<T>,
}

unsafe impl<T: Send + Sync, S: Schedule<ReaderShard<T>>> Sync for ShardedRwLock<T, S> {}
unsafe impl<T: Send + Sync, S: Schedule<ReaderShard<T>>> Send for ShardedRwLock<T, S> {}

pub struct ShardedRwLockHandle<'a, T, S: Schedule<ReaderShard<T>>> {
    shards_handle: BoxedArm<'a, ReaderShard<T>, RWLockScheduler<S>, 1>,
    parent: &'a ShardedRwLock<T, S>,
}

impl<T, S: Schedule<ReaderShard<T>>> ShardedRwLock<T, S>
where
    S: Default,
{
    pub fn new_root(&self) -> ShardedRwLockHandle<'_, T, S> {
        ShardedRwLockHandle {
            shards_handle: self.shards.new_root(),
            parent: self,
        }
    }
}
impl<'a, T, S: Schedule<ReaderShard<T>>> ShardedRwLockHandle<'a, T, S> {
    pub fn read(&mut self) -> Option<ReaderGuard<'a, '_, T>> {
        self.shards_handle
            .offer(LockInput {
                writer: &self.parent.writer,
                data: NonNull::from(&self.parent.item),
                _marker: PhantomData,
            })
            .ok()
    }

    pub fn write(&mut self) -> Option<WriteGuard<'a, T>> {
        self.shards_handle
            .poll(LockInput {
                writer: &self.parent.writer,
                data: NonNull::from(&self.parent.item),
                _marker: PhantomData,
            })
            .ok()
    }

    pub fn fork(&mut self) -> Self {
        Self {
            shards_handle: self.shards_handle.fork(),
            parent: self.parent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    // --- Minimal Mock Scheduler for Testing ---
    #[derive(Default, Debug, Clone, Copy)]
    struct TestScheduler;

    pub struct TestArm;
    impl Hooked for TestArm {
        type State = ();
    }

    impl<T> Schedule<ReaderShard<T>> for TestScheduler {
        type Arm = TestArm;

        fn choose_offer_shard(
            &self,
            _state: &impl StorageBackend<<Self::Arm as Hooked>::State>,
            _arm: &mut Self::Arm,
        ) -> usize {
            0
        }

        fn choose_poll_shard(
            &self,
            _choose_to: &impl StorageBackend<<Self::Arm as Hooked>::State>,
            _arm: &mut Self::Arm,
        ) -> usize {
            0
        }

        fn fork_arm(&self, _arm: &mut Self::Arm) -> Self::Arm {
            TestArm
        }

        fn create_arm(&self) -> Self::Arm {
            TestArm
        }

        fn collect<'b, 'c>(
            &self,
            _state: &impl StorageBackend<<Self::Arm as Hooked>::State>,
            _sub_collections: &'c impl StorageBackend<ReaderShard<T>>,
            _input: <<ReaderShard<T> as Collection>::PollIO as IODescription>::Input<'b>,
        ) -> Option<(
            <<ReaderShard<T> as Collection>::PollIO as IODescription>::Output<'b, 'c>,
            usize,
        )>
        where
            ReaderShard<T>: 'c,
        {
            unreachable!()
        }
    }

    impl<T> ShardedRwLock<T, TestScheduler> {
        pub fn test_new(val: T) -> Self {
            Self {
                shards: BoxedLope::new(8),
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
