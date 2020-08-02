#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "simd"
))]
use faster::*;
use std::ops::{Add, Mul, Sub};
use std::vec::Vec;

pub struct Vector {
    inner: Vec<f64>,
}

impl Add for Vector {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        if self.inner.len() != rhs.inner.len() {
            panic!("Vectors are of different dimensions!");
        }

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "simd"
        ))]
        let inner: Vec<f64> = (
            self.inner.simd_iter(f64s(0.0)),
            rhs.inner.simd_iter(f64s(0.0)),
        )
            .zip()
            .simd_map(|(a, b)| a + b)
            .scalar_collect();

        #[cfg(not(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "simd"
        )))]
        let inner: Vec<f64> = self
            .inner
            .iter()
            .zip(rhs.inner.iter())
            .map(|(a, b)| a + b)
            .collect();

        Vector { inner }
    }
}

impl Sub for Vector {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.inner.len() != rhs.inner.len() {
            panic!("Vectors are of different dimensions");
        }

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "simd"
        ))]
        let inner: Vec<f64> = (
            self.inner.simd_iter(f64s(0.0)),
            rhs.inner.simd_iter(f64s(0.0)),
        )
            .zip()
            .simd_map(|(a, b)| a - b)
            .scalar_collect();

        #[cfg(not(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "simd"
        )))]
        let inner: Vec<f64> = self
            .inner
            .iter()
            .zip(rhs.inner.iter())
            .map(|(a, b)| a - b)
            .collect();

        Vector { inner }
    }
}

impl Mul for Vector {
    type Output = f64;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.inner.len() != rhs.inner.len() {
            panic!("Vectors are of different lenghts");
        }

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "simd"
        ))]
        let total: f64 = (
            self.inner.simd_iter(f64s(0.0)),
            rhs.inner.simd_iter(f64s(0.0)),
        )
            .zip()
            .simd_map(|(a, b)| a * b)
            .sum();

        #[cfg(not(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "simd"
        )))]
        let total: f64 = self
            .inner
            .iter()
            .zip(rhs.inner.iter())
            .map(|(a, b)| a * b)
            .sum();

        total
    }
}

impl Mul for &Vector {
    type Output = f64;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            panic!("Vectors are of different lenghts");
        }

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "simd"
        ))]
        let total: f64 = (
            self.inner.simd_iter(f64s(0.0)),
            rhs.inner.simd_iter(f64s(0.0)),
        )
            .zip()
            .simd_map(|(a, b)| a * b)
            .sum();

        #[cfg(not(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "simd"
        )))]
        let total: f64 = self
            .inner
            .iter()
            .zip(rhs.inner.iter())
            .map(|(a, b)| a * b)
            .sum();

        total
    }
}

impl Mul for &mut Vector {
    type Output = f64;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            panic!("Vectors are of different lenghts");
        }

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "simd"
        ))]
        let total: f64 = (
            self.inner.simd_iter(f64s(0.0)),
            rhs.inner.simd_iter(f64s(0.0)),
        )
            .zip()
            .simd_map(|(a, b)| a * b)
            .sum();

        #[cfg(not(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "simd"
        )))]
        let total: f64 = self
            .inner
            .iter()
            .zip(rhs.inner.iter())
            .map(|(a, b)| a * b)
            .sum();

        total
    }
}

impl Vector {
    pub fn new(length: usize) -> Vector {
        Vector {
            inner: [0.0].repeat(length),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

impl From<Vec<f64>> for Vector {
    fn from(inner: Vec<f64>) -> Self {
        Vector { inner }
    }
}

impl Into<Vec<f64>> for Vector {
    fn into(self) -> Vec<f64> {
        self.inner
    }
}
