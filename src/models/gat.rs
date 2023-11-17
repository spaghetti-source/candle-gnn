use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::{ops, Activation, Init, Module, VarBuilder};

use super::traits::GnnModule;

pub struct GatConv {
    in_dim: usize,
    out_dim: usize,
    num_heads: usize,
    dropout: f32,
    negative_slope: f64,
    weight: Tensor,
    att_src: Tensor,
    att_dst: Tensor,
    activation_fn: Option<Activation>,
}
impl GatConv {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        num_heads: usize,
        negative_slope: f64,
        dropout: f32,
        activation_fn: Option<Activation>,
        vs: VarBuilder,
    ) -> Result<Self> {
        assert!(out_dim % num_heads == 0);
        let hidden_dim = out_dim / num_heads;
        let bound = (6.0 / (in_dim + out_dim) as f64).sqrt();
        Ok(Self {
            in_dim,
            out_dim,
            weight: vs.get_with_hints(
                (in_dim, out_dim),
                "weight",
                Init::Uniform {
                    lo: -bound,
                    up: bound,
                },
            )?,
            att_src: vs.get_with_hints((1, num_heads, hidden_dim), "att_src", Init::Const(0.0))?,
            att_dst: vs.get_with_hints((1, num_heads, hidden_dim), "att_dst", Init::Const(0.0))?,
            num_heads,
            negative_slope,
            dropout,
            activation_fn,
        })
    }
}
impl GnnModule for GatConv {
    fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        assert_eq!(x.shape().rank(), 2);
        assert_eq!(x.shape().dims()[1], self.in_dim);
        let hidden_dim = self.out_dim / self.num_heads;
        let source = edge_index.i((0, ..))?;
        let target = edge_index.i((1, ..))?;

        let num_nodes = x.shape().dims()[0];
        let h = x
            .matmul(&self.weight)?
            .reshape(&[num_nodes, self.num_heads, hidden_dim])?;

        // compute attention
        let attention = {
            let a_src = h.broadcast_mul(&self.att_src)?.sum_keepdim(D::Minus1)?;
            let a_dst = h.broadcast_mul(&self.att_dst)?.sum_keepdim(D::Minus1)?;
            let a_edge = &ops::leaky_relu(
                &(a_src.i(&source)? + a_dst.i(&target)?)?,
                self.negative_slope,
            )?
            .exp()?;
            let a_sum = Tensor::zeros((num_nodes, self.num_heads, 1), x.dtype(), x.device())?
                .index_add(&source, a_edge, 0)?;
            ops::dropout(&a_edge.broadcast_div(&a_sum.i(&source)?)?, self.dropout)
        }?;

        let h = h
            .zeros_like()?
            .index_add(&source, &h.i(&target)?.broadcast_mul(&attention)?, 0)?
            .reshape(&[num_nodes, self.out_dim])?;

        if let Some(a) = self.activation_fn {
            a.forward(&h)
        } else {
            Ok(h)
        }
    }
}
pub struct Gat {
    layers: Vec<GatConv>,
}
impl Gat {
    pub fn new(sizes: &[usize], heads: &[usize], vs: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            let name = format!("layer_{}", i);
            layers.push(GatConv::new(
                sizes[i],
                sizes[i + 1],
                heads[i],
                0.0,
                0.1,
                if i + 1 < sizes.len() {
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
impl GnnModule for Gat {
    fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(&h, edge_index)?;
        }
        Ok(h)
    }
}
