use rand_distr::num_traits::Pow;

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

pub struct Tanh;

impl ActivationFunction for Tanh {
    fn activate(&self, input :f64) -> f64 {
        input.tanh() as f64
    }

    fn derivative(&self, input: f64) -> f64 {
        1.0 - input.tanh().pow(2.0)
    }
}

pub struct ReLU;

impl ActivationFunction for ReLU {
    fn activate(&self, input :f64) -> f64 {
        if input > 0.0 {
            input
        } else {
            0.0
        }
    }

    fn derivative(&self, input: f64) -> f64 {
        if input > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}