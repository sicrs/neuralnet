use std::ops::{ Add, Sub, Mul };
use std::vec::Vec;

pub struct Vector {
    inner: Vec<f64>
}

impl Add for Vector {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        if self.inner.len() != rhs.inner.len() {
            panic!("Vectors are of different dimensions!");
        }

        let mut inner: Vec<f64> = Vec::with_capacity(self.inner.len());
        for i in 0..self.inner.len() {
            inner.push(self.inner[i] + rhs.inner[i]);
        }

        Vector { inner }
        
    }
}

impl Sub for Vector {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.inner.len() != rhs.inner.len() {
            panic!("Vectors are of different dimensions");
        }

        let mut inner: Vec<f64> = Vec::with_capacity(self.inner.len());
        for i in 0..self.inner.len() {
            inner.push(self.inner[i] - rhs.inner[i]);
        }

        Vector { inner }
    }
}

impl Mul for Vector {
    type Output = f64;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.inner.len() != rhs.inner.len() {
            panic!("Vectors are of different lenghts");
        }

        let mut total: f64 = 0.0;
        
        for i in 0..self.inner.len() {
            total += self.inner[i] * rhs.inner[i];
        }

        total
    }
}

impl Vector {
    pub fn new(length: usize) -> Vector {
       Vector {
           inner: [0.0].repeat(length)
       }
    }
}