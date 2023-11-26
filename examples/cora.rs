use anyhow::Result;
use candle_core::{DType, Device, IndexOp, D};
use candle_gnn::datasets::{CoraDataset, FullBatchLoader, RandomSplit};
use candle_gnn::nn::{Gcn, GnnModule};
use candle_nn::loss::cross_entropy;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let dataset = CoraDataset::new("datasets/cora")?;
    let [train_dataset, test_dataset] = dataset.random_split([0.8, 0.2])?;

    let model = Gcn::new(
        &[dataset.num_features(), 16, dataset.num_classes()],
        &device,
    )?;
    let mut optimizer = AdamW::new(model.parameters(), ParamsAdamW::default())?;

    for epoch in 0..200 {
        // training
        let mut train_loss = 0.0;
        let mut train_accuracy = 0.0;
        for batch in FullBatchLoader::new(&train_dataset, &device) {
            let logits = model.forward_t(&batch.xs, &batch.edge_index, true)?;
            let loss = cross_entropy(&logits.i(&batch.mask)?, &batch.ys)?;
            optimizer.backward_step(&loss)?;

            train_loss = loss.to_scalar::<f32>()?;
            train_accuracy = logits
                .i(&batch.mask)?
                .argmax(D::Minus1)?
                .eq(&batch.ys)?
                .to_dtype(DType::F32)?
                .mean_all()?
                .to_scalar::<f32>()?;
        }

        // validation
        if epoch % 10 == 0 {
            for batch in FullBatchLoader::new(&test_dataset, &device) {
                let logits = model.forward(&batch.xs, &batch.edge_index)?;
                let accuracy = logits
                    .i(&batch.mask)?
                    .argmax(D::Minus1)?
                    .eq(&batch.ys)?
                    .to_dtype(DType::F32)?
                    .mean_all()?;
                println!(
                    "{:?} {:?} {:?} {:?}",
                    epoch,
                    train_loss,
                    train_accuracy,
                    accuracy.to_scalar::<f32>()?,
                );
            }
        }
    }
    Ok(())
}
