use candle_core::{Result, Tensor};

pub trait GnnModule {
    fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor>;
}
