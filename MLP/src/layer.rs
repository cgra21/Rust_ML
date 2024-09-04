// Define the Layer object

use crate::neuron::Neuron;

pub struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    pub fn new(num_neurons: usize, num_inputs: usize) -> Layer {
        let neurons: Vec<Neuron> = (0..num_neurons)
            .map(|_| Neuron::new(num_inputs))
            .collect();
        Layer { neurons }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter()
            .map(|neuron| neuron.activate(inputs))
            .collect()
    }
    
    pub fn neurons(&mut self) -> &mut Vec<Neuron> {
        &mut self.neurons
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let num_neurons = 3;
        let num_inputs = 2;
        let layer = Layer::new(num_neurons, num_inputs);
        assert_eq!(layer.neurons.len(), num_neurons);
        for neuron in &layer.neurons {
            assert_eq!(neuron.weights().len(), num_inputs)
        }
    }

    #[test]
    fn test_forward() {
        let num_neurons = 2;
        let num_inputs = 3;
        let mut layer = Layer::new(num_neurons, num_inputs);

        layer.neurons[0].set_weights(vec![0.5, -0.6, 0.2]);
        layer.neurons[0].set_bias(0.1);

        layer.neurons[1].set_weights(vec![0.1, 0.4, -0.3]);
        layer.neurons[1].set_bias(-0.2);

        let inputs = vec![1.0, 2.0, 3.0];
        let outputs = layer.forward(&inputs);

        // Calculate the expected outputs manually
        let expected_output_0: f64 = 1.0 / (1.0 + (-((0.5 * 1.0) + (-0.6 * 2.0) + (0.2 * 3.0) + 0.1) as f64).exp());
        let expected_output_1: f64 = 1.0 / (1.0 + (-((0.1 * 1.0) + (0.4 * 2.0) + (-0.3 * 3.0) - 0.2) as f64).exp());

        let expected_outputs: Vec<f64> = vec![expected_output_0, expected_output_1];

        // Assert that the outputs match the expected outputs
        for (output, expected) in outputs.iter().zip(expected_outputs.iter()) {
            assert!((output - expected).abs() < 1e-6);
        }

        assert_eq!(outputs.len(), num_neurons);

    }
    
    #[test]
    fn test_neurons_mut() {
        let num_neurons = 2;
        let num_inputs = 3;
        let mut layer = Layer::new(num_neurons, num_inputs);
        
        let neurons = layer.neurons();
        assert_eq!(neurons.len(), num_neurons);

        // Modify the first neuron's first weight
        neurons[0].set_weights(vec![42.0, 0.0, 0.0]);
        
        // Check if the modification was successful
        assert_eq!(layer.neurons[0].weights()[0], 42.0);
}
}