use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::{ops, Linear, Module, VarBuilder};

use super::traits::GnnModule;

pub struct GatConv {
    in_dim: usize,
    out_dim: usize,
    num_heads: usize,
    dropout: f32,
    negative_slope: f64,
    lin: Linear,
    att_src: Tensor,
    att_dst: Tensor,
}
impl GatConv {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        num_heads: usize,
        negative_slope: f64,
        dropout: f32,
        vs: VarBuilder,
    ) -> Result<Self> {
        assert!(out_dim % num_heads == 0);
        let hidden_dim = out_dim / num_heads;

        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        Ok(Self {
            in_dim,
            out_dim,
            lin: candle_nn::linear_no_bias(in_dim, out_dim, vs.pp("lin"))?,
            att_src: vs.get_with_hints((1, num_heads, hidden_dim), "att_src", init_ws)?,
            att_dst: vs.get_with_hints((1, num_heads, hidden_dim), "att_dst", init_ws)?,
            num_heads,
            negative_slope,
            dropout,
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
        let h = self
            .lin
            .forward(x)?
            .reshape(&[num_nodes, self.num_heads, hidden_dim])?;

        // compute attention
        let a_src = h.broadcast_mul(&self.att_src)?.sum_keepdim(D::Minus1)?; // (n, h, 1)
        let a_dst = h.broadcast_mul(&self.att_dst)?.sum_keepdim(D::Minus1)?;
        let a_edge = ops::dropout(
            &ops::leaky_relu(
                &(a_src.i(&source)? + a_dst.i(&target)?)?,
                self.negative_slope,
            )?
            .exp()?,
            self.dropout,
        )?;

        let a_sum = Tensor::zeros((num_nodes, self.num_heads, 1), x.dtype(), x.device())?
            .index_add(&source, &a_edge, 0)?;
        let h = h
            .zeros_like()?
            .index_add(&source, &h.i(&target)?.broadcast_mul(&a_edge)?, 0)?
            .broadcast_div(&a_sum)?
            .reshape(&[num_nodes, self.out_dim])?;
        Ok(h)
    }
}
pub struct Gat {
    layers: Vec<GatConv>,
}
impl Gat {
    pub fn new(sizes: &[usize], heads: &[usize], vs: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 1..sizes.len() {
            let name = format!("layer_{}", i);
            layers.push(GatConv::new(
                sizes[i - 1],
                sizes[i],
                heads[i - 1],
                0.0,
                0.1,
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
