use candle_core::{Result, Tensor, IndexOp};
use candle_nn::{Linear, Module, VarBuilder, Activation};

use super::utils::linear;
use super::traits::GnnModule;

struct Mlp {
    fc1: Linear,
    fc2: Linear,
    activation_fn: Activation,
}
impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?.apply(&self.activation_fn)?.apply(&self.fc2)
    }
}
pub struct GinConv {
    nn: Box<dyn Module>,
}
impl GinConv {
    pub fn new(nn: Box<dyn Module>, _vs: VarBuilder) -> Result<Self> {
        Ok(Self { nn })
    }
}
impl GnnModule for GinConv {
    fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let out = x.index_add(
            &edge_index.i((0, ..))?,
            &x.i(&edge_index.i((1, ..))?)?,
            0,
        )?;
        self.nn.forward(&out)
    }
}
pub struct Gin {
    layers: Vec<GinConv>,
}
impl Gin {
    pub fn new(sizes: &[usize], vs: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 1..sizes.len() {
            let name = format!("layer_{}", i);
            let vs_sub = vs.pp(name);
            let mlp = Mlp {
                fc1: linear(sizes[i-1], sizes[i], vs_sub.pp("fc1"))?,
                fc2: linear(sizes[i], sizes[i], vs_sub.pp("fc2"))?,
                activation_fn: Activation::Relu,
            };
            layers.push(GinConv::new(Box::new(mlp), vs_sub)?);
        }
        Ok(Self { layers })
    }
}
impl GnnModule for Gin {
    fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(&h, edge_index)?;
        }
        Ok(h)
    }
}
