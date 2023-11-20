use std::collections::HashMap;

use candle_core::{Result, Tensor};

pub trait GnnModule {
    #[allow(unused_variables)]
    fn forward(&self, xs: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        self.forward_t(xs, edge_index, false)
    }
    fn forward_t(&self, xs: &Tensor, edge_index: &Tensor, train: bool) -> Result<Tensor>;
}

pub trait HeteroGnnModule<NodeType, EdgeType> {
    fn forward(
        &self,
        xs: &HashMap<NodeType, Tensor>,
        edge_index: &HashMap<(NodeType, EdgeType, NodeType), Tensor>,
    ) -> Result<HashMap<NodeType, Tensor>> {
        self.forward_t(xs, edge_index, false)
    }
    fn forward_t(
        &self,
        xs: &HashMap<NodeType, Tensor>,
        edge_index: &HashMap<(NodeType, EdgeType, NodeType), Tensor>,
        train: bool,
    ) -> Result<HashMap<NodeType, Tensor>>;
}
