use std::ops::BitAnd;

use anyhow::Result;
use candle_core::Device;
use polars::{datatypes::Float32Chunked, frame::DataFrame, series::ChunkCompare};

pub trait Dataset {
    type Batch;
    type NodeSelector: Sized;
    fn all_nodes(&self) -> Result<Self::NodeSelector>;
    fn induced_subgraph(&self, nodes: Self::NodeSelector, device: &Device) -> Result<Self::Batch>;
}

pub trait RandomSplit<Ratio> {
    type Output;
    fn random_split(&self, ratio: Ratio) -> Result<Self::Output>;
}

pub trait PolarsDataset {
    fn node_df(&self) -> &DataFrame;
    fn edge_df(&self) -> &DataFrame;
    fn with_node_df(&self, node_df: DataFrame) -> Self;
    fn with_edge_df(&self, edge_df: DataFrame) -> Self;
}
impl<const N: usize, D: PolarsDataset> RandomSplit<[f32; N]> for D {
    type Output = [D; N];
    fn random_split(&self, ratio: [f32; N]) -> Result<Self::Output> {
        let n = self.node_df().height();
        let score = Float32Chunked::rand_uniform("rand", n, 0.0, 1.0);
        let mut cumsum = 0.0;

        let mut result = Vec::new();
        for f in ratio.into_iter() {
            let mask = score.gt_eq(cumsum).bitand(score.lt(cumsum + f));
            cumsum += f;

            let mut node_df = self.node_df().clone();
            node_df.replace_or_add("mask", node_df["mask"].bool()?.bitand(&mask))?;

            result.push(self.with_node_df(node_df));
        }
        match result.try_into() {
            Ok(result) => Ok(result),
            Err(_) => Err(anyhow::anyhow!("err")),
        }
    }
}
