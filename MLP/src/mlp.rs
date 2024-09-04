use crate::layer::Layer;
use itertools::izip;


pub struct MLP {
    layers: Vec<Layer>
}

impl MLP {
    pub fn new(input_size:usize, layer_sizes: &[usize]) -> MLP {
        let mut layers: Vec<Layer> = Vec::new();

        layers.push(Layer::new(layer_sizes[0], input_size));

        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i + 1], layer_sizes[i]));
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
        let output_delta: Vec<f64> = output.iter()
            .zip(targets)
            .map(|(o, t)| o * (1.0 - o) * (t - o))
            .collect();
        deltas.push(output_delta);

        // Calculate hidden layer deltas
        for i in (1..self.layers.len() - 1).rev() {
            let layer: &Layer = &self.layers[i];
            let next_layer: &mut Layer = &mut self.layers[i+1];
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
        for (layer, delta, activation) in izip!(&mut self.layers, &deltas, &activations) {
            for (neuron, delta) in layer.neurons().iter_mut().zip(delta) {
                neuron.update_weights(activation, *delta, learning_rate);
            }
        }


    }
}


#[cfg(test)]
mod tests {
    use std::iter::Sum;

    use super::*;

    #[test]
    fn test_init() {
        let input_size = 3;
        let layer_sizes = vec![5, 2]; // 5 neurons in hidden layer, 2 neurons in output layer

        let mut mlp = MLP::new(input_size, &layer_sizes);

        assert_eq!(mlp.layers.len(), 2);
        assert_eq!(mlp.layers[0].neurons().len(), 5);
        assert_eq!(mlp.layers[1].neurons().len(), 2);

    }

    #[test]
    fn test_forward() {
        let input_size = 3;
        let layer_sizes = vec![5, 2]; // 5 neurons in hidden layer, 2 neurons in output layer

        let mut mlp = MLP::new(input_size, &layer_sizes);

        // Manually set known weights and biases for deterministic output
        for neuron in mlp.layers[0].neurons().iter_mut() {
            neuron.set_weights(vec![0.1, 0.2, 0.3]);
            neuron.set_bias(0.1);
        }

        for neuron in mlp.layers[1].neurons().iter_mut() {
            neuron.set_weights(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
            neuron.set_bias(0.1);
        }

        let inputs = vec![1.0, 2.0, 3.0];
        let activations = mlp.forward(&inputs);

        assert_eq!(activations.len(), 3);

        let expected_output_0 = vec![0.8175745, 0.8175745, 0.8175745, 0.8175745, 0.8175745];

        // Manually calculate the expected output for the second (output) layer
        let expected_output_1: Vec<f64> = expected_output_0
        .iter()
        .map(|&x| 0.1 * x + 0.2 * x + 0.3 * x + 0.4 * x + 0.5 * x + 0.1)
        .map(|z| 1.0 / (1.0 + (-z as f64).exp()))
        .collect();


        assert_eq!(activations[1].len(), 5);
        assert!(activations[1]
            .iter()
            .zip(expected_output_0.iter())
            .all(|(&a, &expected)| (a - expected).abs() < 1e-5)); // Handle rounding errors

        assert_eq!(activations[2].len(), 2);
        assert!(activations[2]
            .iter()
            .zip(expected_output_1.iter())
            .all(|(&a, &expected)| (a - expected).abs() < 1e-5));
        

    }

    #[test]
    fn test_train() {

    }

}