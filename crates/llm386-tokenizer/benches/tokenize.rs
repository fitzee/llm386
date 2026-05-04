//! Token-counting throughput for the cl100k_base adapter across a
//! few realistic input sizes.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llm386_core::Tokenizer;
use llm386_tokenizer::cl100k_base;

fn bench_cl100k(c: &mut Criterion) {
    let tok = cl100k_base().expect("cl100k_base init");
    let short: &[u8] = b"hello world";
    let medium = "lorem ipsum dolor sit amet ".repeat(100); // ~2.7 KB
    let long = "the quick brown fox jumps over the lazy dog. ".repeat(1_000); // ~45 KB

    c.bench_function("cl100k_count_11b", |b| {
        b.iter(|| tok.count(black_box(short)).unwrap());
    });
    c.bench_function("cl100k_count_2_7kb", |b| {
        b.iter(|| tok.count(black_box(medium.as_bytes())).unwrap());
    });
    c.bench_function("cl100k_count_45kb", |b| {
        b.iter(|| tok.count(black_box(long.as_bytes())).unwrap());
    });
}

criterion_group!(benches, bench_cl100k);
criterion_main!(benches);
