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
    
    pub fn neurons(&self) -> &Vec<Neuron> {
        &self.neurons
    }
}
