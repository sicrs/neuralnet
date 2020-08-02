#![allow(dead_code, unused_imports, unused_variables)]
extern crate nalgebra as na;
mod vector;
mod train;
mod source;

#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "simd"))]
use faster::*;
use na::{ Matrix, Dynamic, VecStorage };
use self::vector::Vector;
use self::source::DataSource;

pub enum ActivationFunc {
    Sigmoid,
}

impl ActivationFunc {
    fn get(self) -> Box<dyn Fn(Vector) -> Vector> {
        Box::new(match self {
            _ => |input: Vector| {
                let inner: Vec<f64> = input.into();

                #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "simd"))]
                let res: Vec<f64> = inner
                    .simd_iter(f64s(0.0))
                    .simd_map(|x| f64s(1.0) / (f64s(1.0) + f64s(std::f64::consts::E).powf(- x)))
                    .scalar_collect();

                #[cfg(not(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "simd")))]
                let res: Vec<f64> = inner
                    .iter()
                    .map(|x| 1.0 / (1.0 + std::f64::consts::E.powf(- x)))
                    .collect();

                Vector::from(res)
            }
        })
    }
}

/// The neural network
pub struct Network {
    activation_function: Box<dyn Fn(Vector) -> Vector>,
    bias_matrix: Vec<Vec<f64>>,
    pub configuration: Vec<usize>,
    weight_matrix: Vec<Vec<Vector>>,
}

impl Network {
    pub fn new(configuration: Vec<usize>, activation_function: ActivationFunc) -> Network {
        let n_layers: usize = configuration.len();

        // output neurons do not require biases
        let bias_matrix: Vec<Vec<f64>> = configuration
            .iter()
            .map(|x| [0.0].repeat(*x))
            .collect();
        
        // The input neurons do not require a weight vector
        let weight_matrix: Vec<Vec<Vector>> = configuration[1..]
            .iter()
            .zip(configuration[..(configuration.len() - 1)].iter())
            .map(|(x, y)| {
                let mut v: Vec<Vector> = Vec::with_capacity(*x);
                for i in 0..v.capacity() {
                    v[i] = Vector::new(*y);
                }
                v
            })
            .collect();

        Network { 
            activation_function: activation_function.get(),
            bias_matrix,
            configuration,
            weight_matrix,
        }
    }

    /// Calculate output for the activation func input
    pub fn feedforward(&mut self, layer: usize, neuron: usize, data: &Vector) -> f64 {
        let weight_vec: &Vector = &self.weight_matrix[layer][neuron];
        if data.len() != weight_vec.len() {
            panic!("Vectors are not of the same dimension");
        }

        weight_vec * data + self.bias_matrix[layer][neuron]
    }

    pub fn train<M: train::TrainMethod, D: DataSource>(&mut self, method: M, training_data: D) {
        let ref bias_ref = self.bias_matrix;
        let ref weight_ref = self.weight_matrix;
        if let self::source::DataKind::Training = training_data.kind() {
            
        } else {
            panic!("Non-training data used");
        }
    }
}