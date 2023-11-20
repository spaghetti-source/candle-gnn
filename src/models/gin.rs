use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{
    batch_norm, Activation, BatchNorm, BatchNormConfig, Dropout, Linear, Module, ModuleT,
    VarBuilder,
};

use super::traits::GnnModule;
use super::utils::{linear, sum_agg};

struct Mlp {
    fc1: Linear,
    activation_fn: Activation,
    dropout: Dropout,
    normalization_fn: BatchNorm,
    fc2: Linear,
}
impl Mlp {
    fn new(
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        activation_fn: Activation,
        dropout_rate: f32,
        vs: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            fc1: linear(in_dim, hidden_dim, vs.pp("fc1"))?,
            activation_fn,
            dropout: Dropout::new(dropout_rate),
            fc2: linear(hidden_dim, out_dim, vs.pp("fc2"))?,
            normalization_fn: batch_norm(hidden_dim, BatchNormConfig::default(), vs.pp("bn"))?,
        })
    }
}
impl ModuleT for Mlp {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = self.activation_fn.forward(&xs)?;
        let xs = self.dropout.forward(&xs, train)?;
        let xs = self.normalization_fn.forward(&xs)?;
        let xs = self.fc2.forward(&xs)?;
        Ok(xs)
    }
}
pub struct GinConv {
    nn: Box<dyn ModuleT>,
}
impl GinConv {
    pub fn new(nn: Box<dyn ModuleT>, _vs: VarBuilder) -> Result<Self> {
        Ok(Self { nn })
    }
}
impl GnnModule for GinConv {
    fn forward_t(&self, xs: &Tensor, edge_index: &Tensor, train: bool) -> Result<Tensor> {
        let out = sum_agg(xs, edge_index, xs)?;
        self.nn.forward_t(&out, train)
    }
}

pub struct GinParams {
    activation_fn: Activation,
    dropout_rate: f32,
}
impl Default for GinParams {
    fn default() -> Self {
        Self {
            activation_fn: Activation::Relu,
            dropout_rate: 0.5,
        }
    }
}
pub struct Gin {
    layers: Vec<GinConv>,
}
impl Gin {
    pub fn new(sizes: &[usize], vs: VarBuilder) -> Result<Self> {
        Self::with_params(sizes, GinParams::default(), vs)
    }
    pub fn with_params(sizes: &[usize], params: GinParams, vs: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 1..sizes.len() {
            let name = format!("layer_{}", i);
            let vs_sub = vs.pp(name);
            let mlp = Mlp::new(
                sizes[i - 1],
                sizes[i],
                sizes[i],
                params.activation_fn,
                params.dropout_rate,
                vs_sub.pp("mlp"),
            )?;
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
