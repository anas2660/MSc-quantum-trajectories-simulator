env RUSTFLAGS="-Ctarget-cpu=native -Ctarget-feature=+fma" cargo run --release --features "double-precision"
python plot.py