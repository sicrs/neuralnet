use std::ops::{Add, Mul, Sub};
use super::{Dot, Scale};

pub struct Vector {
    //#[cfg(target_feature = "64bit")]
    inner: Vec<f64>,
    #[cfg(target_feature = "32bit")]
    inner: Vec<f32>,
}

impl Vector {
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    //#[cfg(target_feature = "64bit")]
    pub fn inner_ref(&self) -> &Vec<f64> {
        &self.inner
    }

    #[cfg(target_feature = "32bit")]
    pub fn inner_ref(&self) -> &Vec<f32> {
        &self.inner
    }
}

impl Add for &Vector {
    type Output = Vector;
    fn add(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            panic!("Dimensions do not match!");
        }

        let inner: Vec<_> = self
            .inner
            .iter()
            .zip(rhs.inner.iter())
            .map(|(a, b)| a + b)
            .collect();

        Vector { inner }
    }
}

impl Add for Vector {
    type Output = Vector;
    fn add(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            panic!("Dimensions do not match!");
        }

        let inner: Vec<_> = self
            .inner
            .iter()
            .zip(rhs.inner.iter())
            .map(|(a, b)| a + b)
            .collect();

        Vector { inner }
    }
}

impl Sub for &Vector {
    type Output = Vector;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            panic!("Dimensions do not match!");
        }

        let inner: Vec<_> = self
            .inner
            .iter()
            .zip(rhs.inner.iter())
            .map(|(a, b)| a - b)
            .collect();

        Vector { inner }
    }
}

impl Sub for Vector {
    type Output = Vector;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            panic!("Dimensions do not match!");
        }

        let inner: Vec<_> = self
            .inner
            .iter()
            .zip(rhs.inner.iter())
            .map(|(a, b)| a - b)
            .collect();

        Vector { inner }
    }
}

impl Scale for Vector {
    type Output = Self;
    fn scale(self, factor: f64) -> Self::Output {
        let inner: Vec<f64> = self.into();
        let res: Vec<f64> = inner
            .iter()
            .map(|x| x * factor)
            .collect();
        
        Vector::from(res)
    }
}

impl Scale for &Vector {
    type Output = Vector;
    fn scale(self, factor: f64) -> Self::Output {
        let inner_ref: &Vec<f64> = self.inner_ref();
        let res: Vec<f64> = inner_ref
            .iter()
            .map(|x| x * factor)
            .collect();

        Vector::from(res)
    }
    
}

impl Dot for Vector {
    type Output = f64;
    fn dot(self, rhs: Self) -> Self::Output {
        let total = self
            .inner
            .iter()
            .zip(rhs.inner.iter())
            .map(|(a, b)| a * b)
            .sum();

        total
    }
}

impl Dot for &Vector {
    type Output = f64;
    fn dot(self, rhs: Self) -> Self::Output {
        let total = self
            .inner_ref()
            .iter()
            .zip(rhs.inner.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        total
    }
}

macro_rules! impl_from_to {
    ($t:ty) => {
        impl From<Vec<$t>> for Vector {
            fn from(source: Vec<$t>) -> Self {
                Vector { inner: source }
            }
        }

        impl Into<Vec<$t>> for Vector {
            fn into(self) -> Vec<$t> {
                self.inner
            }
        }
    };
}

//#[cfg(target_feature = "64bit")]
impl_from_to!(f64);
#[cfg(target_feature = "32bit")]
impl_from_to!(f32);
