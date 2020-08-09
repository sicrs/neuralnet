use super::ActivationFunction;
use crate::Vector;

pub struct Sigmoid {}

impl ActivationFunction for Sigmoid {
    fn init() -> Self {
        Sigmoid {}
    }

    fn activation(&self, input: &Vector) -> Vector {
        let inner: &Vec<_> = input.inner_ref();

        let res: Vec<_> = inner
            .iter()
            .map(|x| 1.0 / (1.0 + std::f64::consts::E.powf(-x)))
            .collect();

        Vector::from(res)
    }

    fn derivative(&self, input: &Vector) -> Vector {
        let sigmoid_output = self.activation(input);
        let res: Vec<_> = sigmoid_output
            .inner_ref()
            .iter()
            .map(|x| x * (1.0 - x))
            .collect();

        Vector::from(res)
    }
}
