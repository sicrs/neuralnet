pub mod sigmoid;
use crate::Vector;

pub trait ActivationFunction {
    fn init() -> Self;
    fn activation(&self, input: &Vector) -> Vector;
    fn derivative(&self, input: &Vector) -> Vector;
}