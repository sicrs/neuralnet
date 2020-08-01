#![allow(dead_code, unused_variables)]
extern crate nalgebra as na;
use na::{ Matrix, Dynamic, VecStorage };

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

type InnerMatrix = Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;

pub struct Network {
    activation_function: Box<dyn Fn(f64) -> f64>,
    bias_table: Vec<Vec<f64>>,
    // bias_matrix: InnerMatrix,
    pub configuration: Vec<usize>,
    weight_table: Vec<Vec<f64>>,
    // weight_matrix: InnerMatrix,
    pub layers: usize,
}

impl Network {
    pub fn new(configuration: Vec<usize>, activation_function: ActivationFunc) -> Network {
        let n_layers: usize = configuration.len();
        let mut bias_table: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
        let mut weight_table: Vec<Vec<f64>> = Vec::with_capacity(n_layers);

        // populate the bias and weight tables
        for i in 0..bias_table.capacity() {
            bias_table[i] = Vec::with_capacity(configuration[i]);
        }
        for i in 0..weight_table.capacity() {
            weight_table[i] = Vec::with_capacity(configuration[i]);
        }

        let netw = Network { 
            configuration,
            layers: n_layers,
            bias_table,
            weight_table,
            activation_function: activation_function.get(),
        };
        
        netw
    }
}