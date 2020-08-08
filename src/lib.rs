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

pub enum ActivationFuncKind {
    Sigmoid,
}

struct ActivationFunction {
    kind: ActivationFuncKind,
    activation: Option<Box<dyn Fn(&Vector) -> Vector>>,
    derivative: Option<Box<dyn Fn(&Vector) -> f64>>,
    #[cfg(target_feature = "32bit")]
    derivative: Option<Box<dyn Fn(&Vector) -> f32>>,
}

impl ActivationFunction {
    #[inline(always)]
    fn new(kind: ActivationFuncKind) -> ActivationFunction {
        ActivationFunction {
            kind,
            activation: None,
            derivative: None,
        }
    }

    fn activation(&mut self) -> &Box<dyn Fn(&Vector) -> Vector> {
        if let None = &self.activation {
            self.activation = Some(Box::new(match self.kind {
                _ => |input: &Vector| {
                    //#[cfg(target_feature = "64bit")]
                    let inner: &Vec<f64> = input.inner_ref();
                    #[cfg(target_feature = "32bit")]
                    let inner: &Vec<f32> = input.inner_ref();

                    let res: Vec<_> = inner
                        .iter()
                        .map(|x| 1.0 / (1.0 + std::f64::consts::E.powf(-x)))
                        .collect();

                    Vector::from(res)
                },
            }))
        }

        self.activation.as_ref().unwrap()
    }
}

pub struct Network {
    activation_func: ActivationFunction,
    bias_matrix: Vec<Vec<f64>>,
    pub configuration: &'static [usize],
    weight_matrix: Vec<Vec<Vector>>,
}

impl Network {
    pub fn new(configuration: &'static [usize], activation_func: ActivationFuncKind) -> Network {
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

        assert_eq!(bias_matrix.len(), weight_matrix.len());

        Network {
            activation_func: ActivationFunction::new(activation_func),
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

        (self.activation_func.activation())(&zs_vec)
    }

    pub fn feed(&mut self, input: Vector) -> Vector {
        let mut input_holder: Vector = input;
        for i in 1..self.configuration.len() {
            input_holder = self.direct_feed_layer(&input_holder, i);
        }

        input_holder
    }
}
