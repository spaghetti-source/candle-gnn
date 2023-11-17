use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Activation, Init, Module, VarBuilder};

use super::traits::GnnModule;

pub struct GcnConv {
    weight: Tensor,
    bias: Tensor,
    activation_fn: Option<Activation>,
}
impl GcnConv {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        activation_fn: Option<Activation>,
        vs: VarBuilder,
    ) -> Result<Self> {
        let bound = 1.0 / (in_dim as f64).sqrt();
        let weight = vs.get_with_hints(
            (in_dim, out_dim),
            "weight",
            Init::Uniform {
                lo: -bound,
                up: bound,
            },
        )?;
        let bias = vs.get_with_hints((1, out_dim), "bias", Init::Const(0.0))?;
        Ok(Self {
            weight,
            bias,
            activation_fn,
        })
    }
}
impl GnnModule for GcnConv {
    fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let num_nodes = x.shape().dims()[0];
        let num_edges = edge_index.shape().dims()[1];
        let source = edge_index.i((0, ..))?;
        let target = edge_index.i((1, ..))?;

        let h = x.matmul(&self.weight)?;
        let deg = Tensor::ones((num_nodes, 1), h.dtype(), h.device())?
            .index_add(
                &source,
                &Tensor::ones((num_edges, 1), h.dtype(), h.device())?,
                0,
            )?
            .powf(0.5)?;
        let edge_weight = deg.i(&source)?.mul(&deg.i(&target)?)?;

        let h = h
            .index_add(
                &edge_index.i((0, ..))?,
                &h.i(&target)?.broadcast_div(&edge_weight)?,
                0,
            )?
            .broadcast_add(&self.bias)?;
        if let Some(a) = self.activation_fn {
            a.forward(&h)
        } else {
            Ok(h)
        }
    }
}
pub struct Gcn {
    layers: Vec<GcnConv>,
}
impl Gcn {
    pub fn new(layer_sizes: &[usize], vs: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let name = format!("layer_{}", i);
            layers.push(GcnConv::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                if i + 1 < layer_sizes.len() {
                    Some(Activation::Relu)
                } else {
                    None
                },
                vs.pp(name),
            )?);
        }
        Ok(Self { layers })
    }
}
impl GnnModule for Gcn {
    fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(&h, edge_index)?;
        }
        Ok(h)
    }
}
