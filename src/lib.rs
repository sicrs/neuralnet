#![allow(dead_code, unused_imports, unused_variables)]
mod activation;
mod source;
mod train;
mod vector;

use self::vector::Vector;
use activation::ActivationFunction;

pub struct Network<A: ActivationFunction> {
    activation_func: A,
    bias_matrix: Vec<Vec<f64>>,
    pub configuration: &'static [usize],
    weight_matrix: Vec<Vec<Vector>>,
}

impl<A: ActivationFunction> Network<A> {
    pub fn new(configuration: &'static [usize], activation_func: A) -> Network<A> {
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
            .map(|(neurons, weights)| {
                let mut v: Vec<Vector> = Vec::with_capacity(*neurons);
                for i in 0..*neurons {
                    //#[cfg(target_feature = "64bit")]
                    let inner = [0.0 as f64].repeat(*weights);
                    #[cfg(target_feature = "32bit")]
                    let inner = [0.0 as f32].repeat(*weights);
                    v.push(Vector::from(inner));
                }

                v
            })
            .collect();

        Network {
            activation_func,
            bias_matrix,
            configuration,
            weight_matrix,
        }
    }

    fn direct_feed_layer(&mut self, input: &Vector, layer: usize) -> Vector {
        if layer == 0 {
            panic!("Cannot feed input into input layer");
        }

        if input.len() != self.weight_matrix[layer][0].len() {
            panic!("Dimensions of input vector do not match weight array");
        }

        let zs: Vec<_> = self.weight_matrix[layer]
            .iter()
            .map(|weight_m| weight_m.dot(&input))
            .zip(self.bias_matrix[layer].iter())
            .map(|(wi, bias)| wi + bias)
            .collect();

        let zs_vec = Vector::from(zs);

        self.activation_func.activation(&zs_vec)
    }

    pub fn feed(&mut self, input: Vector) -> Vector {
        let mut input_holder: Vector = input;
        for i in 1..self.configuration.len() {
            input_holder = self.direct_feed_layer(&input_holder, i);
        }

        input_holder
    }
}
