use std::collections::HashMap;
use std::hash::Hash;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Init, Linear, VarBuilder};

//
// Linear layer with torch-equivalent initialisation
//
//   torch.nn.Linear is initialised by Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)).
//   see https://github.com/pytorch/pytorch/issues/57109
//
pub(crate) fn linear(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Linear> {
    let bound = 1.0 / (in_dim as f64).sqrt();
    let init_ws = Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let init_bs = Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bs = vs.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}
#[allow(dead_code)]
pub(crate) fn linear_no_bias(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Linear> {
    let bound = 1.0 / (in_dim as f64).sqrt();
    let init_ws = Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(Linear::new(ws, None))
}

pub fn apply<K, F>(xs: &HashMap<K, Tensor>, f: F) -> Result<HashMap<K, Tensor>>
where
    K: Clone + Hash + Eq,
    F: Fn(&Tensor) -> Result<Tensor>,
{
    xs.iter()
        .map(|(k, x)| f(x).map(|fx| (k.clone(), fx)))
        .collect::<Result<_>>()
}

pub fn out_degree_with_size(edge_index: &Tensor, size: usize) -> Result<Tensor> {
    let dtype = edge_index.dtype();
    let device = edge_index.device();
    Tensor::zeros((size, 1), dtype, device)?.index_add(
        &edge_index.i((0, ..))?,
        &edge_index.i((1, ..))?.ones_like()?,
        0,
    )
}
pub fn out_degree(edge_index: &Tensor) -> Result<Tensor> {
    out_degree_with_size(
        edge_index,
        1 + edge_index.i((0, ..))?.max(D::Minus1)?.to_scalar::<u32>()? as usize,
    )
}
pub fn in_degree_with_size(edge_index: &Tensor, size: usize) -> Result<Tensor> {
    let dtype = edge_index.dtype();
    let device = edge_index.device();
    let n = 1 + edge_index.i((1, ..))?.max(D::Minus1)?.to_scalar::<u32>()? as usize;
    Tensor::zeros((n, 1), dtype, device)?.index_add(
        &edge_index.i((1, ..))?,
        &edge_index.i((0, ..))?.ones_like()?,
        0,
    )
}
pub fn in_degree(edge_index: &Tensor) -> Result<Tensor> {
    in_degree_with_size(
        edge_index,
        1 + edge_index.i((1, ..))?.max(D::Minus1)?.to_scalar::<u32>()? as usize,
    )
}

pub fn mean_agg(xs: &Tensor, edge_index: &Tensor, out: &Tensor) -> Result<Tensor> {
    let dtype = xs.dtype();
    let device = xs.device();
    let n = out.shape().dims()[0];
    let m = edge_index.shape().dims()[1];
    let out_degree = out_degree(edge_index)?.to_dtype(xs.dtype())?;
    out.index_add(
        &edge_index.i((0, ..))?,
        &xs.i(&edge_index.i((1, ..))?)?
            .broadcast_div(&out_degree.i(&edge_index.i((0, ..))?)?)?,
        0,
    )
}
pub fn sum_agg(xs: &Tensor, edge_index: &Tensor, out: &Tensor) -> Result<Tensor> {
    out.index_add(&edge_index.i((0, ..))?, &xs.i(&edge_index.i((1, ..))?)?, 0)
}

pub fn weighted_sum_agg(
    xs: &Tensor,
    edge_index: &Tensor,
    weight: &Tensor,
    out: &Tensor,
) -> Result<Tensor> {
    out.index_add(
        &edge_index.i((0, ..))?,
        &xs.i(&edge_index.i((1, ..))?.broadcast_mul(weight)?)?,
        0,
    )
}
