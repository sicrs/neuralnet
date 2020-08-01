#![allow(dead_code, unused_imports, unused_variables)]
extern crate nalgebra as na;
mod vector;

use na::{ Matrix, Dynamic, VecStorage };
use self::vector::Vector;

pub enum ActivationFunc {
    Sigmoid,
}

impl ActivationFunc {
    fn get(self) -> Box<dyn Fn(f64) -> f64> {
        Box::new(match self {
            _ => |input: f64| {
                1 as f64 / (1 as f64 + std::f64::consts::E.powf(- input))
            }
        })
    }
}

pub struct Network {
    activation_function: Box<dyn Fn(f64) -> f64>,
    bias_matrix: Vec<Vec<f64>>,
    pub configuration: Vec<usize>,
    weight_matrix: Vec<Vec<Vector>>,
}

impl Network {
    pub fn new(configuration: Vec<usize>, activation_function: ActivationFunc) -> Network {
        let n_layers: usize = configuration.len();
        let mut bias_matrix: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
        let mut weight_matrix: Vec<Vec<Vector>> = Vec::with_capacity(n_layers);
        // populate the bias and weight matrices
        for i in 0..bias_matrix.capacity() {
            // the matrix is a table with a bias for each node in the network
            bias_matrix[i] = [0.0].repeat(configuration[i]);
        }

        // populate the weight matrix
        for i in 0..weight_matrix.capacity() {
            for j in 0..configuration[i] {
                // the length of each neuron's weights vector is always the amount of input neurons
                weight_matrix[i][j] = Vector::new(configuration[0]);
            }
        }

        let netw = Network { 
            configuration,
            bias_matrix,
            weight_matrix,
            activation_function: activation_function.get(),
        };
        
        netw
    }
}