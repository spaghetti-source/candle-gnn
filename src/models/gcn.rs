use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Activation, Dropout, Init, Linear, Module, VarBuilder};

use super::{
    traits::GnnModule,
    utils::{in_degree, out_degree, weighted_sum_agg},
};

pub struct GcnConv {
    weight: Tensor,
    bias: Tensor,
}
impl GcnConv {
    pub fn new(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Self> {
        // Xavier Uniform
        let bound = (6.0 / (in_dim + out_dim) as f64).sqrt();
        let weight = vs.get_with_hints(
            (in_dim, out_dim),
            "weight",
            Init::Uniform {
                lo: -bound,
                up: bound,
            },
        )?;
        let bias = vs.get_with_hints((1, out_dim), "bias", Init::Const(0.0))?;
        Ok(Self { weight, bias })
    }
}
impl GnnModule for GcnConv {
    fn forward(&self, xs: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let out_degree = out_degree(edge_index)?;
        let in_degree = in_degree(edge_index)?;
        let edge_weight = out_degree
            .i(&edge_index.i((0, ..))?)?
            .mul(&in_degree.i(&edge_index.i((1, ..))?)?)?
            .to_dtype(xs.dtype())?
            .powf(0.5)?;
        let xs = xs.matmul(&self.weight)?;
        weighted_sum_agg(&xs, edge_index, &edge_weight, &xs)?.broadcast_add(&self.bias)
    }
}
pub struct GcnParams {
    pub dropout_rate: f32,
    pub activation_fn: Activation,
}
impl Default for GcnParams {
    fn default() -> Self {
        Self {
            dropout_rate: 0.0,
            activation_fn: Activation::Relu,
        }
    }
}
pub struct Gcn {
    layers: Vec<GcnConv>,
    dropout: Dropout,
    activation_fn: Activation,
}
impl Gcn {
    pub fn with_params(layer_sizes: &[usize], params: GcnParams, vs: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let name = format!("layer_{}", i);
            layers.push(GcnConv::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                vs.pp(name),
            )?);
        }
        Ok(Self {
            layers,
            dropout: Dropout::new(params.dropout_rate),
            activation_fn: params.activation_fn,
        })
    }
    pub fn new(layer_sizes: &[usize], vs: VarBuilder) -> Result<Self> {
        Self::with_params(layer_sizes, GcnParams::default(), vs)
    }
}
impl GnnModule for Gcn {
    fn forward_t(&self, x: &Tensor, edge_index: &Tensor, train: bool) -> Result<Tensor> {
        let mut h = self.layers[0].forward(x, edge_index)?;
        for layer in &self.layers[1..] {
            h = self.dropout.forward(&h, train)?;
            h = self.activation_fn.forward(&h)?;
            h = layer.forward(&h, edge_index)?;
        }
        Ok(h)
    }
}
