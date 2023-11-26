use super::traits::Dataset;
use candle_core::{DType, Device};

pub struct FullBatchLoader<'a, T> {
    done: bool,
    device: &'a Device,
    dataset: &'a T,
}
impl<'a, T: Dataset + 'a> FullBatchLoader<'a, T> {
    pub fn new(dataset: &'a T, device: &'a Device) -> Self {
        Self {
            done: false,
            device,
            dataset,
        }
    }
}

impl<'a, T: Dataset> Iterator for FullBatchLoader<'a, T> {
    type Item = T::Batch;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            None
        } else {
            let all_nodes = self.dataset.all_nodes().unwrap();
            let batch = self
                .dataset
                .induced_subgraph(all_nodes, self.device)
                .unwrap();
            self.done = true;
            Some(batch)
        }
    }
}
