use candle_core::IndexOp;
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub struct GcnLayer {
    linear: Linear,
}

impl GcnLayer {
    pub fn new(input_dim: usize, output_dim: usize, vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: candle_nn::linear(input_dim, output_dim, vs.pp("linear"))?,
        })
    }
    pub fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let degree = Tensor::ones((x.shape().dims()[0],), x.dtype(), x.device())?;
        let degree = degree.index_add(
            &edge_index.i((0, ..))?,
            &edge_index.i((1, ..))?.ones_like()?.to_dtype(x.dtype())?,
            0,
        )?;
        let degree = degree.reshape((x.shape().dims()[0], 1))?;

        let x = x.zeros_like()?.index_add(
            &edge_index.i((0, ..))?,
            &x.i(&edge_index.i((1, ..))?)?,
            0,
        )?;
        let x = x.broadcast_div(&degree)?;
        let x = self.linear.forward(&x)?;
        let x = x.relu()?;
        Ok(x)
    }
}
pub struct Gcn {
    layers: Vec<GcnLayer>,
}
impl Gcn {
    pub fn new(sizes: &[usize], vs: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 1..sizes.len() {
            let name = format!("layer_{}", i);
            layers.push(GcnLayer::new(sizes[i - 1], sizes[i], vs.pp(name))?);
        }
        Ok(Self { layers })
    }
    pub fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(&h, edge_index)?;
        }
        Ok(h)
    }
}
