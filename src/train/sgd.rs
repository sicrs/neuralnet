use super::Trainer;
use crate::{
    source::DataSource,
    vector::{Dot, Scale},
    ActivationFunction, Network, Vector,
};

pub struct StochasticGradientDescent {
    /// Learning speed
    pub eta: f64,
    /// Length of batch for SGD
    pub subsample_size: usize,
}

impl StochasticGradientDescent {
    pub fn new(eta: f64, subsample_size: usize) -> StochasticGradientDescent {
        StochasticGradientDescent {
            eta,
            subsample_size,
        }
    }
}

impl Trainer for StochasticGradientDescent {
    fn train<A: ActivationFunction, D: DataSource<(Vector, Vector)>>(
        &mut self,
        net: &mut Network<A>,
        data: &mut D,
    ) {
        let num_data_total: usize = data.len();
        let n_subsamples: usize = num_data_total / self.subsample_size;

        // iterate through the amount of subsamples
        for i in 0..n_subsamples {
            // collect self.subsample_size input-output vector tuples in an iterator.
            let subsample_iterator = (0..self.subsample_size).map(|_s| data.next().unwrap());
            // backpropagate
            let results: Vec<(Vec<Vector>, Vec<Vec<Vector>>)> = subsample_iterator
                .map(|io_pair| backpropagate(io_pair, net))
                .collect();
        }
    }
}

fn backpropagate<A: ActivationFunction>(
    io_pair: (Vector, Vector),
    net: &mut Network<A>,
) -> (Vec<Vector>, Vec<Vec<Vector>>) {
    // collect all zs and activations
    // z = w * input + b
    let (input, output) = io_pair;
    let mut zs: Vec<Vector> = Vec::new();
    let mut activations: Vec<Vector> = Vec::new();
    activations.push(input);

    // collect all zs and activations
    for layer in 1..net.configuration.len() {
        let (z, activation) = net.feed_layer(&activations[activations.len() - 1], layer);
        zs.push(z);
        activations.push(activation);
    }

    // declare nabla vars
    // NOTE: The vectors store in reverse
    let mut rev_nabla_bias: Vec<Vector> = Vec::new();
    let mut rev_nabla_weight: Vec<Vec<Vector>> = Vec::new();

    // calculate delta
    // dL / db = (dL / dO) * (dO/db) = (dL / dO) * (dO / dz) * (dz / db) = (dL / dO) * (dO / dz) = delta
    // dL / dw = (dL / dO) * (dO / dz) * (dz / dw) = delta * (dz / dw) = delta * input(layer n)
    rev_nabla_bias.push({
        // calculate delta
        let deriv: Vec<f64> = net.activation_func.derivative(&zs[zs.len() - 1]).into();
        let diff: Vec<f64> = (&activations[activations.len() - 1] - &output).into();
        assert_eq!(deriv.len(), diff.len());
        let delta: Vec<f64> = diff
            .iter()
            .zip(deriv.iter())
            .map(|(df, dv)| df * dv)
            .collect();
        Vector::from(delta)
    });
    rev_nabla_weight.push({
        let delta: &Vec<f64> = rev_nabla_bias[0].inner_ref();
        let mut nabla_ws: Vec<Vector> = Vec::new();
        assert_eq!(delta.len(), (&activations[activations.len() - 2]).len());
        delta
            .iter()
            .for_each(|x| nabla_ws.push((&activations[activations.len() - 2]).scale(*x)));
        nabla_ws
    });

    // from second last layer to first hidden layer
    for layer in 1..(net.configuration.len() - 1) {
        // calculate delta
        rev_nabla_bias.push({
            let sigmoid_prime: Vector = net.activation_func.derivative(&zs[zs.len() - (layer + 1)]);
            let weights: &Vec<Vector> = &net.weight_matrix[net.weight_matrix.len() - layer];
            let prev_delta = &rev_nabla_bias[rev_nabla_bias.len() - 1];
            let dot_prod: Vec<f64> = weights.iter().map(|x| x.dot(prev_delta)).collect();

            assert_eq!(sigmoid_prime.inner_ref().len(), dot_prod.len());
            let nabla_b: Vec<f64> = sigmoid_prime
                .inner_ref()
                .iter()
                .zip(dot_prod.iter())
                .map(|(sp, dp)| sp * dp)
                .collect();

            Vector::from(nabla_b)
        });

        rev_nabla_weight.push({
            let delta = &rev_nabla_bias[rev_nabla_bias.len() - 1];
            let activation = &activations[activations.len() - (layer + 1)];
            let nabla_w: Vec<Vector> = delta
                .inner_ref()
                .iter()
                .map(|x| activation.scale(*x))
                .collect();

            nabla_w
        })
    }

    let nabla_bias: Vec<Vector> = rev_nabla_bias.into_iter().rev().collect();
    let nabla_weight: Vec<Vec<Vector>> = rev_nabla_weight.into_iter().rev().collect();

    (nabla_bias, nabla_weight)
}
