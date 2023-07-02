$env:RUST_BACKTRACE=1
cargo build --release

chdir ecc_py
maturin develop


