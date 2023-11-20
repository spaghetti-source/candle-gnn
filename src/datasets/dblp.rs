use candle_core::{DType, Device, Tensor};
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NodeType {
    Author,
    Paper,
    Term,
    Conference,
}
impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Author => write!(f, "author"),
            Self::Paper => write!(f, "paper"),
            Self::Term => write!(f, "term"),
            Self::Conference => write!(f, "conference"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EdgeType {
    To,
}
impl std::fmt::Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::To => write!(f, "to"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Dblp {
    pub x: std::collections::HashMap<NodeType, Tensor>,
    pub y: Tensor,
    pub train_mask: Tensor,
    pub val_mask: Tensor,
    pub test_mask: Tensor,
    pub edge_index: std::collections::HashMap<(NodeType, EdgeType, NodeType), Tensor>,
    // pub node_types: [NodeType; 4],
    // pub edge_types: [(NodeType, EdgeType, NodeType); 6],
}
impl Dblp {
    /// As the conference nodes have no features, we add one-hot vectors.
    pub fn from_file<P: AsRef<Path>>(path: P, device: &Device) -> anyhow::Result<Self> {
        let mut hashmap: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::from_iter(Tensor::read_npz(path.as_ref())?);
        let x = std::collections::HashMap::from([
            (
                NodeType::Author,
                hashmap.remove("author_x").unwrap().to_device(device)?,
            ),
            (
                NodeType::Paper,
                hashmap.remove("paper_x").unwrap().to_device(device)?,
            ),
            (
                NodeType::Term,
                hashmap.remove("term_x").unwrap().to_device(device)?,
            ),
            (NodeType::Conference, Tensor::eye(20, DType::F32, device)?),
        ]);
        let y = hashmap
            .remove("author_y")
            .unwrap()
            .to_dtype(DType::U32)?
            .to_device(device)?;
        let train_mask = hashmap
            .remove("author_train_mask")
            .unwrap()
            .to_device(device)?;
        let val_mask = hashmap
            .remove("author_val_mask")
            .unwrap()
            .to_device(device)?;
        let test_mask = hashmap
            .remove("author_test_mask")
            .unwrap()
            .to_device(device)?;

        let edge_index = std::collections::HashMap::from([
            (
                (NodeType::Author, EdgeType::To, NodeType::Paper),
                hashmap
                    .remove("author_to_paper_edge_index")
                    .unwrap()
                    .to_device(device)?,
            ),
            (
                (NodeType::Paper, EdgeType::To, NodeType::Author),
                hashmap
                    .remove("paper_to_author_edge_index")
                    .unwrap()
                    .to_device(device)?,
            ),
            (
                (NodeType::Paper, EdgeType::To, NodeType::Term),
                hashmap
                    .remove("paper_to_term_edge_index")
                    .unwrap()
                    .to_device(device)?,
            ),
            (
                (NodeType::Term, EdgeType::To, NodeType::Paper),
                hashmap
                    .remove("term_to_paper_edge_index")
                    .unwrap()
                    .to_device(device)?,
            ),
            (
                (NodeType::Paper, EdgeType::To, NodeType::Conference),
                hashmap
                    .remove("paper_to_conference_edge_index")
                    .unwrap()
                    .to_device(device)?,
            ),
            (
                (NodeType::Conference, EdgeType::To, NodeType::Paper),
                hashmap
                    .remove("conference_to_paper_edge_index")
                    .unwrap()
                    .to_device(device)?,
            ),
        ]);
        Ok(Dblp {
            x,
            y,
            train_mask,
            val_mask,
            test_mask,
            edge_index,
        })
    }
}
