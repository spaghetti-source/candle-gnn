use candle_core::{Tensor, Result};

pub fn mask_to_index(mask: &Tensor) -> Result<Tensor> {
    Tensor::from_iter(
        mask.to_vec1()?
            .into_iter()
            .enumerate()
            .filter_map(|(idx, m): (_, u8)| if m == 0 { None } else { Some(idx as u32) } ),
        mask.device(),
    )
}
