use super::Trainer;
use crate::{source::DataSource, ActivationFunction, Network, Vector};

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
            subsample_iterator.for_each(|io_pair| backpropagate(io_pair, net));
        }
    }
}

fn backpropagate<A: ActivationFunction>(io_pair: (Vector, Vector), net: &mut Network<A>) {
    // collect all zs and activations
    // z = w * input + b
    let (input, output) = io_pair;
    let mut zs: Vec<Vector> = Vec::new();
    let mut activations: Vec<Vector> = Vec::new();
    activations.push(input);

    // collect all zs and activations
    for layer in 0..net.configuration.len() {
        let (z, activation) = net.feed_layer(&activations[activations.len() - 1], layer);
        zs.push(z);
        activations.push(activation);
    }

    // declare nabla vars
    // NOTE: The vectors store in reverse
    let mut rev_nabla_bias: Vec<Vector> = Vec::new();
    let mut rev_nabla_weight: Vec<Vec<Vector>> = Vec::new();

    rev_nabla_bias.push({
        // calculate delta
        let deriv: Vec<f64> = net.activation_func.derivative(&zs[zs.len() - 1]).into();
        let diff: Vec<f64> = (&activations[activations.len() - 1] - &output).into();
        let delta: Vec<f64> = diff
            .iter()
            .zip(deriv.iter())
            .map(|(df, dv)| df * dv)
            .collect();
        Vector::from(delta)
    });
}
