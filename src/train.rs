use crate::{source::DataSource, Network, Vector};
use faster::*;

pub trait Trainer {
    fn train<D: DataSource<(Vector, Vector)>>(&mut self, net: &mut Network, data: D);
}

pub struct StochasticGradientDescent {
    /// Learning speed
    pub eta: f64,
    /// Length of batch used for Stochastic Gradient Descent (SGD)
    pub subsample_size: usize,
}

impl StochasticGradientDescent {
    pub fn new(eta: f64, subsample_size: usize) -> StochasticGradientDescent {
        StochasticGradientDescent {
            eta,
            subsample_size,
        }
    }

    fn train_inner<D>(&mut self, net: &mut Network, data: D)
    where
        D: DataSource<(Vector, Vector)>,
    {
        let dataset_collect: Vec<&(Vector, Vector)> = data.iter().collect();
        let iterations = dataset_collect.len() / self.subsample_size;

        // iterate through samples and break them into subgroups of length self.subsample_size
        let subsample_iterator = (0..iterations).map(|x| {
            Vec::from(&dataset_collect[(x * self.subsample_size)..((x + 1) * self.subsample_size)])
        });

        // iterate through subgroups and update the once every run through a subgroup
        for subsample in subsample_iterator {
            let mut counter: usize = 0;

            // iterate through the subgroup and calculate the partial derivative of the cost function

            // iterate through samples in the subsample and generate activation and loss func derivatives
            let activation = subsample.iter().map(|(input, output)| {
                
                if counter == 0 {
                    let inner: Vec<f64> = input.inner_ref().iter().map(|x| *x).collect();
                    Vector::from(inner)
                } else {
                    counter += 1;
                    net.feed_layer(input, counter)
                }
            });
        }

        todo!();
    }

    fn backpropagate(&self) {
        todo!();
    }
}

impl Trainer for StochasticGradientDescent {
    fn train<D: DataSource<(Vector, Vector)>>(&mut self, net: &mut Network, data: D) {
        self.train_inner(net, data);
    }
}
