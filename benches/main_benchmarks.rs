use std::{
    hint::black_box,
    sync::{Arc, Barrier, RwLock as StdRwLock},
    thread,
    time::{Duration, Instant},
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

#[allow(unused_imports)]
use lope::schedule::{Hooked, RandomAccess, RoundRobin, Schedule};
use shardedrw::ShardedRwLock;

const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8];
// 0 = all writes, 100 = all reads
const READ_PERCENTAGES: &[u8] = &[0, 50, 90, 99, 100];
const OPS_PER_THREAD: usize = 20_000;

// ---------------------------------------------------------------------
// Workload: run `n_threads`, each doing OPS_PER_THREAD ops mixed
// according to `read_pct`, and return the wall-clock duration of the
// timed region (post-barrier-release to all-joined).
// ---------------------------------------------------------------------

type BenchedScheduler = RandomAccess<SmallRng>;
const SHARDS: usize = 8;

fn run_sharded(n_threads: usize, read_pct: u8) -> Duration {
    let lock: ShardedRwLock<u64, BenchedScheduler> = ShardedRwLock::new(SHARDS, 0);
    let mut root = lock.new_root();

    let barrier = Barrier::new(n_threads + 1);

    let start = thread::scope(|s| {
        for i in 0..n_threads {
            let barrier = &barrier;
            let mut handle = root.fork();
            let seed = i as u64;

            s.spawn(move || {
                let mut rng = SmallRng::seed_from_u64(seed);
                barrier.wait();
                for _ in 0..OPS_PER_THREAD {
                    if rng.random_range(0..100) < read_pct {
                        loop {
                            if let Some(guard) = handle.read() {
                                black_box(&*guard);
                                break;
                            }
                            thread::yield_now();
                        }
                    } else {
                        loop {
                            if let Some(mut guard) = handle.write() {
                                *guard = guard.wrapping_add(1);
                                break;
                            }
                            thread::yield_now();
                        }
                    }
                }
            });
        }

        // Release all worker threads from the barrier simultaneously
        barrier.wait();
        Instant::now()
    }); // Implicitly joins all scoped threads here before exiting

    start.elapsed()
}

fn run_std(n_threads: usize, read_pct: u8) -> Duration {
    let lock = Arc::new(StdRwLock::new(0u64));

    let barrier = Arc::new(Barrier::new(n_threads + 1));
    let handles: Vec<_> = (0..n_threads)
        .map(|i| {
            let barrier = Arc::clone(&barrier);
            let lock = Arc::clone(&lock);
            let seed = i as u64;
            thread::spawn(move || {
                let mut rng = SmallRng::seed_from_u64(seed);
                barrier.wait();
                for _ in 0..OPS_PER_THREAD {
                    if rng.random_range(0..100) < read_pct {
                        loop {
                            if let Ok(guard) = lock.try_read() {
                                black_box(&*guard);
                                break;
                            }
                            thread::yield_now();
                        }
                    } else {
                        loop {
                            if let Ok(mut guard) = lock.try_write() {
                                *guard = guard.wrapping_add(1);
                                break;
                            }
                            thread::yield_now();
                        }
                    }
                }
            })
        })
        .collect();

    barrier.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    start.elapsed()
}

// ---------------------------------------------------------------------
// Criterion wiring
// ---------------------------------------------------------------------

fn bench_contention_matrix(c: &mut Criterion) {
    for &read_pct in READ_PERCENTAGES {
        let mut group = c.benchmark_group(format!("read_pct_{read_pct}"));
        for &n_threads in THREAD_COUNTS {
            group.throughput(Throughput::Elements((n_threads * OPS_PER_THREAD) as u64));

            group.bench_with_input(
                BenchmarkId::new("sharded_rwlock", n_threads),
                &n_threads,
                |b, &n_threads| {
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            total += run_sharded(n_threads, read_pct);
                        }
                        total
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("std_rwlock", n_threads),
                &n_threads,
                |b, &n_threads| {
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            total += run_std(n_threads, read_pct);
                        }
                        total
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    // Contention benchmarks are noisy and slow (real thread spawns per
    // sample) — fewer samples, longer measurement time than defaults.
    config = Criterion::default().sample_size(30).measurement_time(Duration::from_secs(5));
    targets = bench_contention_matrix
}
criterion_main!(benches);
