use candle_core::DType;
use candle_core::{Device, Tensor};
use std::path::Path;

use std::collections::HashMap;

// ```python
// dataset = dataset = Planetoid(root='data/Planetoid', name='Cora')
// data = {
//     "x": dataset.x.numpy(),
//     "edge_index": dataset.edge_index.numpy(),
//     "y": dataset.y.numpy(),
//     "train_indices": np.where(dataset.train_mask)[0],
//     "val_indices": np.where(dataset.val_mask)[0],
//     "test_indices": np.where(dataset.test_mask)[0],
// }
// with open("cora.npz", "wb") as f:
//     np.savez(f, **data)
// ```
pub struct Cora {
    pub x: Tensor,
    pub edge_index: Tensor,
    pub y: Tensor,
    pub train_indices: Tensor,
    pub val_indices: Tensor,
    pub test_indices: Tensor,
}
impl Cora {
    pub fn from_file<P: AsRef<Path>>(cora_npz: P, device: &Device) -> anyhow::Result<Self> {
        let mut hashmap: HashMap<String, Tensor> =
            HashMap::from_iter(Tensor::read_npz(cora_npz.as_ref())?);
        Ok(Self {
            x: hashmap.remove("x").unwrap().to_device(device)?,
            edge_index: hashmap
                .remove("edge_index")
                .unwrap()
                .to_dtype(DType::U32)?
                .to_device(device)?,
            y: hashmap
                .remove("y")
                .unwrap()
                .to_dtype(DType::U32)?
                .to_device(device)?,
            train_indices: hashmap
                .remove("train_indices")
                .unwrap()
                .to_dtype(DType::U32)?
                .to_device(device)?,
            val_indices: hashmap
                .remove("val_indices")
                .unwrap()
                .to_dtype(DType::U32)?
                .to_device(device)?,
            test_indices: hashmap
                .remove("test_indices")
                .unwrap()
                .to_dtype(DType::U32)?
                .to_device(device)?,
        })
    }
}
