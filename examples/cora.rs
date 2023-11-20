#![allow(unused_imports)]
#![allow(dead_code)]

use candle_core::{DType, Device, IndexOp, D, Result, Tensor};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap, Activation, Module, Dropout};

use candle_gnn::datasets::Cora;
use candle_gnn::nn::{Gat, Gcn, Gin, GnnModule};

fn train_evaluate<G: GnnModule>(cora: &Cora, gnn: G, varmap: VarMap, name: &str) -> anyhow::Result<()> {
    let num_epochs = 100;

    let mut optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: 0.01,
            weight_decay: 0.0005,
            ..Default::default()
        },
    )?;

    for epoch in 0..num_epochs {
        let logits = gnn.forward_t(&cora.x, &cora.edge_index, true)?;
        let loss = loss::cross_entropy(&(logits.i(&cora.train_indices)?), &(cora.y.i(&cora.train_indices)?))?;
        optimizer.backward_step(&loss)?;

        if epoch % 1 == 0 {
            let logits = gnn.forward_t(&cora.x, &cora.edge_index, false)?;

            let is_ok = logits.argmax(D::Minus1)?.eq(&cora.y)?.to_dtype(DType::F32)?;
            let train_accuracy = is_ok.i(&cora.train_indices)?.mean_all()?.to_scalar::<f32>()?;
            let val_accuracy = is_ok.i(&cora.val_indices)?.mean_all()?.to_scalar::<f32>()?;
            let test_accuracy = is_ok.i(&cora.test_indices)?.mean_all()?.to_scalar::<f32>()?;
            println!(
                "[{}] Epoch: {epoch:3} Train loss: {:8.5} Train accuracy {:5.2}% Val accuracy {:5.2}% Test accuracy: {:5.2}%",
                name,
                loss.to_scalar::<f32>()?,
                100.0 * train_accuracy,
                100.0 * val_accuracy,
                100.0 * test_accuracy,
            );
        }
    }
    println!("--------");
    Ok(())
}

// cargo run --example cora
fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;

    let cora = Cora::from_file("datasets/cora.npz", &device)?;
    /*
    if false {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = Gcn::new(&[1433, 16, 7], vs.pp("gcn"))?;
        train_evaluate(&cora, model, varmap, "GCN")?;
    }
    if true {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let gnn = Gin::new(&[1433, 16, 7], vs.pp("gin"))?;
        train_evaluate(&cora, gnn, varmap, "GIN")?;
    }
    */
    if true {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let gnn = Gat::new(&[1433, 16, 7], &[4, 1], vs.pp("gat"))?;
        train_evaluate(&cora, gnn, varmap, "GAT")?;
    }
    Ok(())
}
