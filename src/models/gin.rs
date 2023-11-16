use candle_core::{Result, Tensor, IndexOp};
use candle_nn::{Linear, Module, VarBuilder};

use super::traits::GnnModule;

struct Mlp {
    fc1: Linear,
    activation_fn: candle_nn::Activation,
    fc2: Linear,
}
impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?.apply(&self.activation_fn)?.apply(&self.fc2)
    }
}
pub struct GinConv {
    nn: Box<dyn Module>,
    eps: Tensor,
}
impl GinConv {
    pub fn new(nn: Box<dyn Module>, vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            nn,
            eps: vs.get_with_hints((1,), "eps", candle_nn::init::Init::Const(0.0))?,
        })
    }
}
impl GnnModule for GinConv {
    fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let x = x.broadcast_mul(&(1.0 + &self.eps)?)?.index_add(
            &edge_index.i((0, ..))?,
            &x.i(&edge_index.i((1, ..))?)?,
            0,
        )?;
        self.nn.forward(&x)
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
            let vs = vs.pp(name);
            let hidden_dim = (sizes[i] + sizes[i-1]) / 2;
            let mlp = Mlp {
                fc1: candle_nn::linear(sizes[i-1], hidden_dim, vs.pp("fc1"))?,
                activation_fn: candle_nn::Activation::Relu,
                fc2: candle_nn::linear(hidden_dim, sizes[i], vs.pp("fc2"))?,
            };
            layers.push(GinConv::new(Box::new(mlp), vs)?);
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

