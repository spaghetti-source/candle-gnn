use anyhow::Result;
use candle_core::Device;

pub trait Dataset {
    type Batch;
    type NodeSelector: Sized;
    fn all_nodes(&self) -> Result<Self::NodeSelector>;
    fn induced_subgraph(&self, nodes: Self::NodeSelector, device: &Device) -> Result<Self::Batch>;
}

pub trait NewFromDataset<T> {
    fn new(dataset: T, device: Device) -> Self;
}

pub trait RandomSplit<Ratio> {
    type Output;
    fn random_split(&self, ratio: Ratio) -> Result<Self::Output>;
}
