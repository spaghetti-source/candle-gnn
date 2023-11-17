use candle_core::{Result};
use candle_nn::{Linear, VarBuilder, Init};

//
// Linear layer with torch-equivalent initialisation
//
//   torch.nn.Linear is initialised by Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)).
//   see https://github.com/pytorch/pytorch/issues/57109
//
pub(crate) fn linear(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Linear> {
    let bound = 1.0 / (in_dim as f64).sqrt();
    let init_ws = Init::Uniform { lo: -bound, up: bound };
    let init_bs = Init::Uniform { lo: -bound, up: bound };
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bs = vs.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}
#[allow(dead_code)]
pub(crate) fn linear_no_bias(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Linear> {
    let bound = 1.0 / (in_dim as f64).sqrt();
    let init_ws = Init::Uniform { lo: -bound, up: bound };
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(Linear::new(ws, None))
}
