pub mod sgd;

use crate::{source::DataSource, ActivationFunction, Network, Vector};

pub trait Trainer {
    fn train<A: ActivationFunction, D: DataSource<(Vector, Vector)>>(
        &mut self,
        net: &mut Network<A>,
        data: &mut D,
    );
}
