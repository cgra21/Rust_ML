pub trait ActivationFunction {
    fn activate(&self, input :f64) -> f64;
    fn derivative(&self, input: f64) -> f64;
}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn activate(&self, input: f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
    }

    fn derivative(&self, input: f64) -> f64 {
        let sigmoid = self.activate(input);
        sigmoid * (1.0 - sigmoid)
    }
}