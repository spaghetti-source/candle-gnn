use candle_core::{DType, Device, IndexOp, D};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};

use candle_gnn::datasets::Cora;
use candle_gnn::models::Gcn;

// cargo run --example cora
fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // load dataset
    let cora = Cora::from_file("datasets/cora.npz", &device)?;

    // create a GCN model
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let gcn = Gcn::new(&[1433, 128, 7], vs.pp("gcn"))?;

    // training loop
    let mut optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            ..Default::default()
        },
    )?;
    for epoch in 0..100 {
        let logits = gcn.forward(&cora.x, &cora.edge_index)?;
        let loss = loss::cross_entropy(
            &(logits.i(&cora.train_indices)?),
            &(cora.y.i(&cora.train_indices)?),
        )?;
        optimizer.backward_step(&loss)?;

        if epoch % 5 == 0 {
            let logits = gcn.forward(&cora.x, &cora.edge_index)?;

            let is_ok = logits
                .argmax(D::Minus1)?
                .eq(&cora.y)?
                .to_dtype(DType::F32)?;
            let train_accuracy = is_ok
                .i(&cora.train_indices)?
                .mean_all()?
                .to_scalar::<f32>()?;
            let val_accuracy = is_ok.i(&cora.val_indices)?.mean_all()?.to_scalar::<f32>()?;
            let test_accuracy = is_ok
                .i(&cora.test_indices)?
                .mean_all()?
                .to_scalar::<f32>()?;
            println!(
                "Epoch: {epoch:3} Train loss: {:8.5} Train accuracy {:5.2}% Val accuracy {:5.2}% Test accuracy: {:5.2}%",
                loss.to_scalar::<f32>()?,
                100.0 * train_accuracy,
                100.0 * val_accuracy,
                100.0 * test_accuracy,
            );
        }
    }
    Ok(())
}
