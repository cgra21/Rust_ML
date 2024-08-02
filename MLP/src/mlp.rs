use crate::layer::Layer;
use rand::Rng;


pub struct MLP {
    layers: Vec<Layer>
}

impl MLP {
    pub fn new(layer_sizes: &[usize]) -> MLP {
        let mut layers: Vec<Layer> = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_size[i + 1], layer_sizes[i]));
        }
        MLP { layers }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut activations: Vec<Vec<f64>> = vec![inputs.to_vec()];
        for layer in &self.layers {
            let input: &Vec<f64> = activations.last().unwrap();
            let output: Vec<f64> = layer.forward(input);
            activations.push(output);
        }
        activations
    }

    pub fn train(&mut self, inputs: &[f64], targets: &[f64], learning_rate: f64) {
        // Forward pass
        let activations: Vec<Vec<f64>> = self.forward(inputs);

        // Calculate output layer delta
        let mut deltas: Vec<Vec<f64>> = Vec::new();
        let output = activations.last().unwrap();
        let output_delta: Vec<f64> = output.ier()
            .zip(targets)
            .map(|o, t| o * (1.0 - o) * (t - o))
            .collect();
        deltas.push(output_delta);

        // Calculate hidden layer deltas
        for i in (1..self.layers.len() - 1).rev() {
            let layer: &Layer = &self.layers[i];
            let next_layer: &Layer = &self.layers[i+1];
            let next_delta: &Vec<f64> = &deltas[0];
            let activation: &Vec<f64> = &activations[i];

            let delta: Vec<f64> = activation.iter()
                .enumerate()
                .map(|(j, a)| {
                    let sum: f64 = next_layer.neurons().iter()
                        .map(|neuron| neuron.weights()[j] * next_delta[j])
                        .sum();
                    a * (1.0 - a) * sum
                })
                .collect();
            deltas.insert(0, delta);
        
        }

        // Update weights and biass
        for (layer, delta, ativation) in izip!(&mut self.layers, &deltas, &activations) {
            for (neuron, delta) in layer.neurons.iter_mut().zip(delta) {
                neuron.update_weights(activation, *delta, learning_rate);
            }
        }


    }
}
