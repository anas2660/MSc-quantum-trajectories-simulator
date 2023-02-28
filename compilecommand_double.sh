RUSTFLAGS="-Ctarget-cpu=native -Ctarget-feature=+fma" mold -run cargo run --release --features "double-precision"
