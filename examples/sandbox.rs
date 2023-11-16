use candle_core::{DType, Device, IndexOp, D};
use candle_nn::{VarBuilder, VarMap};

use candle_gnn::datasets::Cora;
use candle_gnn::models::Gat;

// cargo run --example sandbox
fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // load dataset
    let cora = Cora::from_file("datasets/cora.npz", &device)?;

    // create a GCN model
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let gat = Gat::new(
        &[1433, 128, 7], &[8, 1],
        vs.pp("gat"),
    )?;
    let h = gat.forward(
        &cora.x,
        &cora.edge_index,
    );
    println!("{:?}", h);
    Ok(())
}
