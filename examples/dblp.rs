/*
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear, loss, ops, Init, Linear, Optimizer, VarBuilder, VarMap};

use candle_gnn::datasets::{Dblp, EdgeType, NodeType};
use candle_gnn::nn::{hetero_gcn, HeteroGcnConv, HeteroGnnModule, utils::apply};
use candle_gnn::utils::mask_to_index;

use EdgeType::*;
use NodeType::*;

use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;

struct DblpBaselineModel {
    lin1: Linear,
    lin2: Linear,
}
impl DblpBaselineModel {
    fn new(vs: VarBuilder) -> Result<Self> {
        let hidden_dim = 16;
        let num_classes = 4;
        Ok(Self {
            lin1: linear(334, hidden_dim, vs.pp("lin1"))?,
            lin2: linear(hidden_dim, num_classes, vs.pp("lin2"))?,
        })
    }
    fn forward(&self, dblp: &Dblp, train: bool) -> Result<Tensor> {
        let mut xs = self.lin1.forward(&dblp.x[&Author])?;
        xs = xs.relu()?;
        if train {
            xs = ops::dropout(&xs, 0.5)?;
        }
        xs = self.lin2.forward(&xs)?;
        Ok(xs)
    }
}

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let dblp = Dblp::from_file("datasets/dblp.npz", &device)?;

    // hack until Candle implements mask
    let train_index = mask_to_index(&dblp.train_mask)?;
    let test_index = mask_to_index(&dblp.test_mask)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let num_classes = 4;
    let hidden_dim = 128;
    let model = hetero_gcn(
        &[
            &[(Author, 334), (Paper, 4231), (Term, 50), (Conference, 20)],
            &[
                (Author, hidden_dim),
                (Paper, hidden_dim),
                (Term, hidden_dim),
                (Conference, hidden_dim),
            ],
            &[(Author, num_classes)],
        ],
        &[
            (Author, To, Paper),
            (Paper, To, Author),
            (Paper, To, Term),
            (Term, To, Paper),
            (Paper, To, Conference),
            (Conference, To, Paper),
        ],
        vs,
    )?;
    // let model = DblpModel::new(vs)?;
    // let model = DblpBaselineModel::new(vs)?;

    let mut optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            ..Default::default()
        },
    )?;

    for epoch in 0..100 {
        let logits = model
            .forward_t(&dblp.x, &dblp.edge_index, true)?
            .remove(&Author)
            .unwrap();
        let loss = loss::cross_entropy(&(logits.i(&train_index)?), &(dblp.y.i(&train_index)?))?;
        optimizer.backward_step(&loss)?;

        let logits = model
            .forward_t(&dblp.x, &dblp.edge_index, false)?
            .remove(&Author)
            .unwrap();
        let is_ok = logits
            .argmax(D::Minus1)?
            .eq(&dblp.y)?
            .to_dtype(DType::F32)?;
        let train_accuracy = is_ok.i(&train_index)?.mean_all()?.to_scalar::<f32>()?;
        let test_accuracy = is_ok.i(&test_index)?.mean_all()?.to_scalar::<f32>()?;
        println!(
            "epoch={} train_loss={}; train_acc={}; test_acc={}",
            epoch,
            loss.to_scalar::<f32>()?,
            train_accuracy,
            test_accuracy
        );
    }

    Ok(())
}

 */
 fn main() { }