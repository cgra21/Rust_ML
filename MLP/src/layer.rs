// Define the Layer object
use crate::activation::ActivationFunction;
use ndarray::{Array2, Array1};
use std::any::Any;
use rand_distr::{Uniform, Distribution};
use rand::thread_rng;

pub trait Layer {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64>;
    fn backward(&mut self, delta: &Array1<f64>, learning_rate: f64) -> Array1<f64>;

    fn as_any(&self) -> &dyn Any;
}

// Fully Connected Layer
pub struct FCLayer {
    weights: Array2<f64>,
    bias: Array1<f64>,
    input: Option<Array1<f64>>, // Store input for backprop
}

impl FCLayer {
    pub fn new(num_inputs: usize, num_outputs: usize) -> FCLayer {
        let mut rng = thread_rng();
        // We will intiailize the weights with a uniform Xaiver initialization
        let bound = (6.0 / (num_inputs + num_outputs) as f64).sqrt();
        let uniform_dist = Uniform::from(-bound..bound);

        let weights = Array2::from_shape_fn((num_outputs, num_inputs), |_| uniform_dist.sample(&mut rng));
        let bias = Array1::zeros(num_outputs); // Initialize bias vector with zeros
        FCLayer { weights, bias, input: None}
    }

    // Getter for weights
    pub fn get_weights(&self) -> &Array2<f64> {
        &self.weights
    }

    // Getter for bias
    pub fn get_bias(&self) -> &Array1<f64> {
        &self.bias
    }
}

impl Layer for FCLayer {
    fn forward(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        self.input = Some(inputs.clone());
        
        // println!("Input shape: {:?}", inputs.shape());
        // println!("Weights shape: {:?}", self.weights.shape());
        self.weights.dot(inputs) + &self.bias

        }

    fn backward(&mut self, delta: &Array1<f64>, learning_rate: f64) -> Array1<f64>{
        let input = self.input.as_ref().expect("Input should be set in forward pass");

        let input_error = self.weights.t().dot(delta);

        // Reshape input to be a column vector (n, 1) and delta to be a row vector (1, m)
        let input_col = input.to_shape((1, input.len())).unwrap();  // Shape (1, num_inputs)
        let delta_row = delta.to_shape((delta.len(), 1)).unwrap();  // Shape (num_outputs, 1) 

        // Compute the gradient: X^T . delta
        let weights_error = delta_row.dot(&input_col); // (num_inputs, num_outputs)

        // println!("Input shape: {:?}", input.shape());
        // println!("Delta shape: {:?}", delta.shape());
        // println!("Weights shape: {:?}", self.weights.shape());
        // println!("Weights Error shape: {:?}", weights_error.shape());

        assert_eq!(self.weights.shape(), weights_error.shape());

        self.weights -= &(weights_error.mapv(|w| learning_rate * w));
        self.bias -= &(delta.mapv(|d| learning_rate * d));

        input_error

        }

        fn as_any(&self) -> &dyn Any {
            self
        }

    }

// Activation Layer 
pub struct ActivationLayer {
    activation: Box<dyn ActivationFunction>,
    input: Option<Array1<f64>>,
}

impl ActivationLayer {
    pub fn new(activation: Box<dyn ActivationFunction>) -> ActivationLayer {
        ActivationLayer {
            activation,
            input: None,        
        }
    }
}

impl Layer for ActivationLayer {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.input = Some(input.clone());

        input.mapv(|x| self.activation.activate(x))
    }

    fn backward(&mut self, delta: &Array1<f64>, _learning_rate: f64) -> Array1<f64> {
        let input = self.input.as_ref().expect("Input should be set in forward pass");

        input.mapv(|x| self.activation.derivative(x)) * delta
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Conv2D Layer
pub enum Padding {
    Valid, // Only convolutes on existiing data, no padding
    Same, // Ensures that input will stay the same size in output
    Explicit(usize, usize), // Explicit values (height, width)
}

pub struct Conv2D {
    input_channels: usize, // Input features, i.e. RGB has 3 channels
    output_channels: usize,
    kernel_size: (usize, usize), // (kernel_height, kernel_width)
    stride: (usize, usize), // (stride_height, stride_width)
    padding: Padding,
    weights: Array4<f64>, // 4D, (output_channels, input_channels, kernel_height, kernel_width)
    // ex. (5, 3, 3, 3)
    // [, 
    //     [
    //      [[1, 2, 3], [1, 2, 3], [1, 2, 3]], Weights for R channel
    //      [[1, 2, 3], [1, 2, 3], [1, 2, 3]], Weights for B channel
    //      [[1, 2, 3], [1, 2, 3], [1, 2, 3]], Weights for G channel
    //     ], This is a single output channel, this will repeat 5 times, each input channel has a 3x3 kernel
    bias: Array1<f64>,
}

impl Conv2D {
    pub fn new(
        input_channels: usize,
        output_channels: usize,
        kernel_size: (usize, usize),
        
    ) {

    }
}

// impl Layer for Conv2D {
//     fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        
//     }

//     fn backward(&mut self, delta: &Array1<f64>, learning_rate: f64) -> Array1<f64> {
        
//     }

//     fn as_any(&self) -> &dyn Any {
//         self
//     }
// }




#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Sigmoid;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_fc_layer_forward() {
        let mut fc_layer = FCLayer {
            weights: arr2(&[[0.1, 0.2], [0.3, 0.4]]),
            bias: arr1(&[0.1, 0.2]),
            input: None,
        };

        let input = arr1(&[1.0, 2.0]);

        let output = fc_layer.forward(&input);

        let expected_output = arr1(&[
            0.1 * 1.0 + 0.2 * 2.0 + 0.1, 
            0.3 * 1.0 + 0.4 * 2.0 + 0.2
        ]);
        assert_eq!(output, expected_output);


    }

    #[test]
    fn test_fc_layer_backward() {
        let mut fc_layer = FCLayer {
            weights: arr2(&[[0.1, 0.2], [0.3, 0.4]]),
            bias: arr1(&[0.1, 0.2]),
            input: None,
        };

        // Need to perform forward pass to update inputs
        let input = arr1(&[1.0, 2.0]);

        let output = fc_layer.forward(&input);

        let expected_output = arr1(&[0.1 * 1.0 + 0.2 * 2.0 + 0.1, 
                                                                        0.3 * 1.0 + 0.4 * 2.0 + 0.2]);
        assert_eq!(output, expected_output);

        let delta = arr1(&[0.1, 0.2]);
        let learning_rate = 0.01;

        let input_error = fc_layer.backward(&delta, learning_rate);

        let expected_input_error = arr1(&[
            0.1 * 0.1 + 0.3 * 0.2,
            0.2 * 0.1 + 0.4 * 0.2,
        ]);

        assert_eq!(input_error, expected_input_error);

        let expected_updated_weights = arr2(&[
            [0.1 - 0.01 * 1.0 * 0.1, 0.2 - 0.01 * 2.0 * 0.1],
            [0.3 - 0.01 * 1.0 * 0.2, 0.4 - 0.01 * 2.0 * 0.2],
        ]);

        assert_eq!(fc_layer.weights, expected_updated_weights);

        let expected_bias = arr1(&[
            0.1 - 0.01 * 0.1,
            0.2 - 0.01 * 0.2,
        ]);

        assert_eq!(fc_layer.bias, expected_bias);

    }

    #[test]
    fn test_activation_forward() {
        let mut activation = ActivationLayer {
            activation: Box::new(Sigmoid),
            input: None,
        };

        let input = arr1(&[1.0, 2.0]);

        let expected_output = arr1(&[
            (1.0 / (1.0 + (-1.0 as f64).exp())),
            (1.0 / (1.0 + (-2.0 as f64).exp())),
        ]);

        let output = activation.forward(&input);

        assert_eq!(output, expected_output);

    }

    #[test]
    fn test_activation_backward() {
        let mut activation = ActivationLayer {
            activation: Box::new(Sigmoid),
            input: None,
        };

        let input = arr1(&[1.0, 2.0]);

        let expected_output = arr1(&[
            (1.0 / (1.0 + (-1.0 as f64).exp())),
            (1.0 / (1.0 + (-2.0 as f64).exp())),
        ]);

        // Need to perform forward first in order to store input
        let output = activation.forward(&input);

        assert_eq!(output, expected_output);

        let delta = arr1(&[0.1, 0.2]);
        let learning_rate = 0.01; // This doesn't matter since there are no learnable parameters in this layer

        let expected_input_error = arr1(&[
            // (sigmoid * (1.0 - sigmoid)) * delta:
            (1.0 / (1.0 + (-1.0 as f64).exp())) * (1.0 - (1.0 / (1.0 + (-1.0 as f64).exp()))) * delta[0],
            (1.0 / (1.0 + (-2.0 as f64).exp())) * (1.0 - (1.0 / (1.0 + (-2.0 as f64).exp()))) * delta[1],
        ]);

        let input_error = activation.backward(&delta, learning_rate);

        assert_eq!(input_error, expected_input_error); 
    }

}