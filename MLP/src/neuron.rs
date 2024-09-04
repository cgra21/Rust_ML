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

    // Getters
    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }

    // Setters
    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let neuron = Neuron::new(3);
        assert_eq!(neuron.weights.len(), 3);
        assert_eq!(neuron.bias, 0.0);
    }
    
    #[test]
    fn test_activation() {
        let mut neuron = Neuron::new(3);
        neuron.weights = vec![0.5, -0.6, 0.2];
        neuron.bias = 0.1;
    
        let inputs = vec![1.0, 2.0, 3.0];
        let output = neuron.activate(&inputs);
        
        let expected_output = 0.5; // Manually calculate the expected output
        assert!((output - expected_output).abs() < 1e-6); // Allow for floating-point precision errors
    }

    #[test]
    fn test_update_weights() {
        let mut neuron = Neuron::new(2);
        neuron.weights = vec![0.5, -0.5];
        neuron.bias = 0.1;

        let inputs = vec![1.0, -1.0];
        let delta = 0.5;
        let learning_rate = 0.1;

        neuron.update_weights(&inputs, delta, learning_rate);

        let expected_weights = vec![0.5 + 0.1 * 0.5 * 1.0, -0.5 + 0.1 * 0.5 * -1.0];
        let expected_bias = 0.1 + 0.1 * 0.5;
    
        assert_eq!(neuron.weights, expected_weights);
        assert!((neuron.bias - expected_bias).abs() < 1e-6);
    }
}