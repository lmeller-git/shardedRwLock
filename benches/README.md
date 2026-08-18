# Benchmarks

## Tests

- `single thread`: `N` items are pushed and popped sequentially from a queue.
- `spsc`: One thread pushes N items. Another thread pops `N` items.
- `mpsc`: `T` threads push `N / T` items each. Another thread pops `N` items.
- `mpmc`: `T` threads push `N / T` items each. `M` other threads pop `N` total items.

`spsc` test is done for multiple `N`.
`mpmc` test is done for multiple queue capacities.

To run `mpmc` test across multiple capacities, feature `alloc` must be enabled.
To bench `crossbeam_queue::ArrayQueue`, `RUSTFLAGS=--cfg bench_crossbeam` and feature `alloc` must be provided.


## Running

To run all benchmarks:

```bash
RUSTFLAGS="--cfg bench_crossbeam" cargo bench --all-features
```

## Results

TODO: add main results (+ machine info, rust version, ...)
