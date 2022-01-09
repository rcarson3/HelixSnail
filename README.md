# HelixSnail
A small nonlinear solver library written in Rust. It is based on the c++ [snls](https://github.com/LLNL/SNLS) library by LLNL. This crate is largely an experiment to see how we can make performant code in a safer language where we don't have to resort to pointer filled code. Additionally, this code is also generic over the float type which the original one was only suitable to run with f64/double types.

It is written such that it can be used in `no_std` environments, so it can be used on the GPU using crates such as [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA).
