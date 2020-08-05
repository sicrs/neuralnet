#![allow(dead_code, unused_imports, unused_variables)]
mod source;
mod train;
mod vector;

use self::vector::Vector;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "simd"
))]
use faster::*;
use source::DataSource;

pub enum ActivationFunc {
    Sigmoid,
}

impl ActivationFunc {
    fn get(self) -> Box<dyn Fn(Vector) -> Vector> {
        Box::new(match self {
            _ => |input: Vector| {
                let inner: Vec<f64> = input.into();

                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    target_feature = "simd"
                ))]
                let res: Vec<f64> = inner
                    .simd_iter(f64s(0.0))
                    .simd_map(|x| f64s(1.0) / (f64s(1.0) + f64s(std::f64::consts::E).powf(-x)))
                    .scalar_collect();

                #[cfg(not(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    target_feature = "simd"
                )))]
                let res: Vec<f64> = inner
                    .iter()
                    .map(|x| 1.0 / (1.0 + std::f64::consts::E.powf(-x)))
                    .collect();

                Vector::from(res)
            },
        })
    }

    fn prime(self) -> Box<dyn Fn(Vector) -> f64> {
        Box::new(match self {
            _ => |input: Vector| {
                let func = self.get();
                func(input) * (Vector::from(Vec::from(&([1.0].repeat(input.len())))) - func(input))
            },
        })
    }
}

pub struct NetworkInner {
    activation_func: ActivationFunc,
    bias_matrix: Vec<Vec<f64>>,
    pub configuration: &'static [usize],
    weight_matrix: Vec<Vec<Vector>>,
}

impl NetworkInner {
    pub fn new(configuration: &'static [usize], activation_func: ActivationFunc) -> NetworkInner {
        let n_layers: usize = configuration.len();

        // bias matrix - initial bias of 0
        let bias_matrix: Vec<Vec<f64>> = configuration[1..]
            .iter()
            .map(|x| [0.0].repeat(*x))
            .collect();

        // weight matrix
        let weight_matrix: Vec<Vec<Vector>> = configuration[1..]
            .iter()
            .zip(configuration[..(configuration.len() - 1)].iter())
            .map(|(x, y)| {
                let mut v: Vec<Vector> = Vec::with_capacity(*x);
                for i in 0..v.capacity() {
                    v.push(Vector::new(*y));
                }
                v
            })
            .collect();

        NetworkInner {
            activation_func,
            bias_matrix,
            configuration,
            weight_matrix,
        }
    }

    fn feed_layer(&self, input: Vector, layer: usize) -> Vector {
        if layer == 0 {
            panic!("Cannot feed into input layer!");
        }

        if input.len() != self.weight_matrix[layer][0].len() {
            panic!("Dimensions of input vector do not match weight matrix");
        }

        let weigh_prod_collection: Vec<f64> = self.weight_matrix[layer]
            .iter()
            .zip(self.bias_matrix[layer].iter())
            .map(|(weight, bias)| weight * &input + bias)
            .collect();


        todo!();
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
    pub fn new(configuration: &[usize], activation_function: ActivationFunc) -> Network {
        let configuration: Vec<usize> = Vec::from(configuration);
        let n_layers: usize = configuration.len();

        // output neurons do not require biases
        let bias_matrix: Vec<Vec<f64>> = configuration.iter().map(|x| [0.0].repeat(*x)).collect();

        // The input neurons do not require a weight vector
        // NOTE: index 0 contains the weights for the nodes of layer 2 (index layer 1)
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

    /// Middleware function
    fn feed_layer(&self, data: &Vector, layer: usize) -> Vector {
        if layer == 0 {
            panic!("The first layer does not have weights");
        }

        if data.len() != self.weight_matrix[layer][0].len() {
            panic!("The dimension of the input vector does not correspond with the dimension of the weight vector");
        }

        let weight_prod_collection: Vec<f64> = self.weight_matrix[layer - 1]
            .iter()
            .map(|weight| weight * &data)
            .collect();

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "simd"
        ))]
        let result: Vec<f64> = (
            weight_prod_collection.simd_iter(f64s(0.0)),
            self.bias_matrix[layer].simd_iter(f64s(0.0)),
        )
            .zip()
            .simd_map(|(weight_prod, bias)| weight_prod + bias)
            .scalar_collect();

        #[cfg(not(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "simd"
        )))]
        let result: Vec<f64> = weight_prod_collection
            .iter()
            .zip(self.bias_matrix[layer].iter())
            .map(|(weight_prod, bias)| weight_prod + bias)
            .collect();

        Vector::from(result)
    }

    /// Calculate output for the activation func input
    pub fn feed(&self, input: Vector) -> Vector {
        let n_layers = self.configuration.len();
        let mut layer_input: Vector = input;

        for i in 1..n_layers {
            layer_input = self.feed_layer(&layer_input, i);
        }

        layer_input
    }

    pub fn train<T: train::Trainer, D: DataSource<(Vector, Vector)>>(
        &mut self,
        trainer: &mut T,
        training_data: D,
    ) {
        trainer.train(self, training_data);
    }
}
