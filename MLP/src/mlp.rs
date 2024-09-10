use crate::layer::Layer;
use ndarray::Array1;

pub struct MLP {
    layers: Vec<Box<dyn Layer>>,
}

impl MLP {

    pub fn new() -> MLP {
        MLP {
            layers: Vec::new(), // Start with no layers
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn remove_layer(&mut self) {
        self.layers.pop();
    }
    
    pub fn forward(&mut self, input: &ndarray::Array1<f64>) -> ndarray::Array1<f64> {
        let mut current_input = input.clone();
        //println!("Initial input shape: {:?}", current_input.shape());
        for layer in self.layers.iter_mut() {
            current_input = layer.forward(&current_input);
            //println!("Layer output shape: {:?}", current_input.shape());
        }
        current_input
    }

    pub fn backward(&mut self, output_delta: &ndarray::Array1<f64>, learning_rate: f64) {
        let mut current_delta = output_delta.clone();
        for layer in self.layers.iter_mut().rev() {
            current_delta = layer.backward(&current_delta, learning_rate);
        }
    }

    pub fn train(
        &mut self, 
        input: &ndarray::Array1<f64>, 
        target: &ndarray::Array1<f64>, 
        learning_rate: f64,
        epochs: usize,
    ) {
        for epoch in 0..epochs {
            let output = self.forward(input);

            let delta = &output - target;

            self.backward(&delta, learning_rate);

            // Optionally print the loss
            let epoch_loss = delta.mapv(|x| x.powi(2)).sum(); // Mean Square Error
            println!("Epoch {}: Loss = {:.6}", epoch + 1, epoch_loss);
        }

    }

    pub fn train_batch(
        &mut self, 
        inputs: &Vec<Array1<f64>>, 
        targets: &Vec<Array1<f64>>, 
        learning_rate: f64, 
        epochs: usize
    ) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
    
            // Accumulate the gradient over all samples before applying updates
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let output = self.forward(input);
                let delta = &output - target;
    
                total_loss += delta.mapv(|x| x.powi(2)).sum();  // Mean Square Error
    
                // Perform backward pass (adjusting weights) after every input
                self.backward(&delta, learning_rate);
            }
    
            // Print loss every 1000 epochs
            if epoch % 1000 == 0 {
                println!("Epoch {}: Total Loss = {:.6}", epoch, total_loss);
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{FCLayer, ActivationLayer};  // Assuming these are defined elsewhere
    use crate::activation::Sigmoid;
    use ndarray::arr1;

    #[test]
    fn test_add_layer() {
        let mut mlp = MLP::new();

        mlp.add_layer(Box::new(FCLayer::new(2,2)));
        mlp.add_layer(Box::new(ActivationLayer::new(Box::new(Sigmoid))));

        assert_eq!(mlp.layers.len(), 2);
    }
    
    #[test]
    fn test_remove_layer() {
        let mut mlp = MLP::new();

        mlp.add_layer(Box::new(FCLayer::new(2,2)));
        mlp.add_layer(Box::new(ActivationLayer::new(Box::new(Sigmoid))));
        mlp.add_layer(Box::new(FCLayer::new(2, 1)));

        mlp.remove_layer();

        assert_eq!(mlp.layers.len(), 2);
    }

    #[test]
    fn test_train() {
        let mut mlp = MLP::new();
    
        // Add layers
        mlp.add_layer(Box::new(FCLayer::new(2, 2)));  // 2 input neurons, 2 hidden neurons
        mlp.add_layer(Box::new(ActivationLayer::new(Box::new(Sigmoid))));
        mlp.add_layer(Box::new(FCLayer::new(2, 1)));  // 2 hidden neurons, 1 output neuron
    
        // Input and target
        let input = arr1(&[0.5, 0.8]);
        let target = arr1(&[0.2]);
    
        // Perform training step
        let learning_rate = 0.01;
        let epochs = 100;
    
        // Track loss to ensure it's decreasing
        let mut previous_loss = f64::MAX;
    
        for epoch in 0..epochs {
            let output = mlp.forward(&input);
            let delta = &output - &target;
            let current_loss = delta.mapv(|x| x.powi(2)).sum(); // Mean Square Error
    
            // Print the loss, weights, and biases for debugging
            println!("Epoch {}: Loss = {:.6}", epoch + 1, current_loss);
    
            let weights_layer_1 = mlp.layers[0].as_any().downcast_ref::<FCLayer>().unwrap().get_weights();
            let bias_layer_1 = mlp.layers[0].as_any().downcast_ref::<FCLayer>().unwrap().get_bias();
            let weights_layer_2 = mlp.layers[2].as_any().downcast_ref::<FCLayer>().unwrap().get_weights();
            let bias_layer_2 = mlp.layers[2].as_any().downcast_ref::<FCLayer>().unwrap().get_bias();
    
            println!("Weights Layer 1: {:?}", weights_layer_1);
            println!("Bias Layer 1: {:?}", bias_layer_1);
            println!("Weights Layer 2: {:?}", weights_layer_2);
            println!("Bias Layer 2: {:?}", bias_layer_2);
    
            // Check if the loss is decreasing (not required but a good sanity check)
            assert!(current_loss < previous_loss, "Loss did not decrease at epoch {}", epoch + 1);
            previous_loss = current_loss;
    
            // Update the model using backward propagation
            mlp.backward(&delta, learning_rate);
        }
    }

    #[test]
    fn test_train_xor_batch() {
        let mut mlp = MLP::new();

        // Add layers
        mlp.add_layer(Box::new(FCLayer::new(2, 3)));  // 2 input neurons, 2 hidden neurons
        mlp.add_layer(Box::new(ActivationLayer::new(Box::new(Sigmoid))));
        mlp.add_layer(Box::new(FCLayer::new(3, 1)));  // 2 hidden neurons, 1 output neuron
        mlp.add_layer(Box::new(ActivationLayer::new(Box::new(Sigmoid))));

        // XOR input and target data
        let inputs = vec![
            ndarray::Array1::from_vec(vec![0.0, 0.0]),
            ndarray::Array1::from_vec(vec![0.0, 1.0]),
            ndarray::Array1::from_vec(vec![1.0, 0.0]),
            ndarray::Array1::from_vec(vec![1.0, 1.0]),
        ];

        let targets = vec![
            ndarray::Array1::from_vec(vec![0.0]),  // XOR(0, 0) = 0
            ndarray::Array1::from_vec(vec![1.0]),  // XOR(0, 1) = 1
            ndarray::Array1::from_vec(vec![1.0]),  // XOR(1, 0) = 1
            ndarray::Array1::from_vec(vec![0.0]),  // XOR(1, 1) = 0
        ];

        // Training parameters
        let learning_rate = 0.1;
        let epochs = 10000;

        // Train the network with batch updates
        mlp.train_batch(&inputs, &targets, learning_rate, epochs);

        // Test the network after training
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = mlp.forward(input);
            println!("Input: {:?}, Target: {:?}, Output: {:?}", input, target, output);
        }
    }

}