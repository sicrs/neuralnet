/*#[cfg(not(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "simd"
)))]*/
mod vector;
/*#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "simd"
))]*/
//mod vector_simd;

pub use vector::*;
