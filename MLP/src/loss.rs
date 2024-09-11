pub trait LossFunction {
    fn activate(&self, input :f64) -> f64;
    fn derivative(&self, input: f64) -> f64;
}

pub struct MSE;

impl LossFunction for MSE {
    fn activate(&self, target: &Array1<f64>, predicted: &Array1<f64>) -> f64{
        let diff = predicted - target;
        diff.mapv(|x| x.powi(2)).mean().unwrap()
    }

    fn derivative(&self, target: &Array1<f64>, predicted: &Array1<f64>) -> Array1<f64> {
        2.0 * (predicted - target) / predicted.len()
    }

}