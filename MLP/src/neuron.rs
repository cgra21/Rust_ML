pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    pub fn new(num_inputs:usize) -> Neuron {
        let weights: Vec<f64> = vec![0.0; num_inputs]; // Initialize weights
        let bias: f64 = 0.0; // Initialize bias to zero
        Neuron { weights, bias }
    }

    pub fn activate(&self, inputs: &[f64]) -> f64 {
        let sum: f64 = self.weights.iter()
                                   .zip(inputs)
                                   .map(|(w, i)| w * i)
                                   .sum::<f64>() + self.bias;
        1.0 / (1.0 + (-sum).exp()) // Sigmoid function
    }

    pub fn update_weights(&mut self, inputs: &[f64], delta: f64, learning_rate: f64) {
        for (w, i) in self.weights.iter_mut().zip(inputs) {
            *w += learning_rate * delta * i;
        }
        self.bias += learning_rate * delta;
    }

    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }
}