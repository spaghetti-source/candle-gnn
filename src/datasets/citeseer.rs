use std::{
    fs::File,
    io::{BufRead, BufReader},
    ops::BitAnd,
    path::Path,
};

use ::zip::ZipArchive;
use anyhow::Result;
use candle_core::{Device, Tensor};
use polars::prelude::*;

use super::{traits::Dataset, RandomSplit};
use super::{utils::RemoteFile, EdgeDirection};

#[derive(Debug, Clone)]
pub struct CiteSeerBatch {
    pub xs: Tensor,
    pub edge_index: Tensor,
    pub ys: Tensor,
    pub mask: Tensor, // loss(&logits.i(mask)?, &ys)
}

#[derive(Debug, Clone)]
pub struct CiteSeerDataset {
    pub num_features: usize,
    pub num_classes: usize,
    node_df: DataFrame,
    feature_cols: Vec<String>,
    edge_df: DataFrame,
}
impl CiteSeerDataset {
    pub fn new<P: AsRef<Path>>(root: P) -> anyhow::Result<Self> {
        let root = root.as_ref();
        if !root.exists() {
            Self::download(root)?;
        }
        Self::load(root, EdgeDirection::default())
    }
    pub fn download<P: AsRef<Path>>(root: P) -> anyhow::Result<()> {
        // download file
        let mut local_file = tempfile::tempfile()?;
        let mut remote_file =
            RemoteFile::with_pbar("https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.zip")?;
        std::io::copy(&mut remote_file, &mut local_file)?;

        // unzip
        let path = root.as_ref().join("raw");
        std::fs::create_dir_all(&path)?;

        let mut archive = ZipArchive::new(local_file)?;
        for name in ["citeseer/citeseer.content", "citeseer/citeseer.cites"] {
            let filename = name.split('/').last().unwrap();
            std::io::copy(
                &mut archive.by_name(name)?,
                &mut File::create(path.join(filename))?,
            )?;
        }
        Ok(())
    }
    pub fn load<P: AsRef<Path>>(root: P, edge_direction: EdgeDirection) -> anyhow::Result<Self> {
        // read endpoints
        let reader = BufReader::new(File::open(root.as_ref().join("raw").join("cora.cites"))?);
        let mut source = Vec::new();
        let mut target = Vec::new();
        for buf in reader.lines() {
            let line = buf?;
            let mut iter = line.split_whitespace();
            let u: u32 = iter.next().unwrap().parse()?;
            let v: u32 = iter.next().unwrap().parse()?;

            if edge_direction.has_forward_edges() {
                source.push(u);
                target.push(v);
            }
            if edge_direction.has_reverse_edges() {
                source.push(v);
                target.push(u);
            }
            assert!(iter.next().is_none());
        }
        let edge_df = df! {
            "source" => source,
            "target" => target,
        }?;

        // read features and labels
        let reader = BufReader::new(File::open(root.as_ref().join("raw").join("cora.content"))?);
        let mut id = Vec::new();
        let mut xs = vec![Vec::new(); 1433];
        let mut label = Vec::new();

        for buf in reader.lines() {
            let line = buf?;
            let mut iter = line.split_whitespace();
            let u: u32 = iter.next().unwrap().parse()?;
            for xs_i in xs.iter_mut() {
                let x: f32 = iter.next().unwrap().parse()?;
                xs_i.push(x);
            }
            let y = iter.next().unwrap().to_owned();
            id.push(u);
            label.push(y);
            assert!(iter.next().is_none());
        }
        let mut node_df = df! {
            "id" => id.clone(),
            "label" => label,
            "mask" => vec![true; id.len()],
        }?;
        let mut feature_cols = Vec::new();
        for (i, x) in xs.into_iter().enumerate() {
            let name = format!("xs[{}]", i);
            node_df.with_column(Series::from_vec(&name, x))?;
            feature_cols.push(name);
        }
        let label =
            df! { "label" => node_df["label"].unique()? }?.with_row_count("label_u32", None)?;
        let node_df = node_df.inner_join(&label, ["label"], ["label"])?;
        Ok(Self {
            num_features: 1433,
            num_classes: 7,
            node_df,
            edge_df,
            feature_cols,
        })
    }
    pub fn node_df(&self) -> &DataFrame {
        &self.node_df
    }
    pub fn edge_df(&self) -> &DataFrame {
        &self.edge_df
    }
    fn id_cols(&self) -> &[&str] {
        &["id"]
    }
}

impl<const N: usize> RandomSplit<[f32; N]> for CiteSeerDataset {
    type Output = [CiteSeerDataset; N];
    fn random_split(&self, ratio: [f32; N]) -> Result<Self::Output> {
        let n = self.node_df.height();
        let score = Float32Chunked::rand_uniform("rand", n, 0.0, 1.0);
        let mut cumsum = 0.0;

        let mut result = Vec::new();
        for f in ratio.into_iter() {
            let mask = score.gt_eq(cumsum).bitand(score.lt(cumsum + f));
            cumsum += f;

            let mut node_df = self.node_df.clone();
            node_df.replace_or_add("mask", node_df["mask"].bool()?.bitand(&mask))?;
            result.push(Self {
                node_df,
                edge_df: self.edge_df.clone(),
                num_features: self.num_features,
                num_classes: self.num_classes,
                feature_cols: self.feature_cols.clone(),
            });
        }
        match result.try_into() {
            Ok(result) => Ok(result),
            Err(e) => Err(anyhow::anyhow!("err")),
        }
    }
}

impl Dataset for CiteSeerDataset {
    type Batch = CiteSeerBatch;
    type NodeSelector = DataFrame;

    fn all_nodes(&self) -> Result<DataFrame> {
        let result = self.node_df.select(self.id_cols())?;
        Ok(result)
    }
    fn induced_subgraph(&self, nodes: DataFrame, device: &Device) -> Result<CiteSeerBatch> {
        let index = nodes.with_row_count("__index", None)?;

        let node_df = index.inner_join(&self.node_df, ["id"], ["id"])?;
        let mut xs = Vec::new();
        for col in node_df.select_series(&self.feature_cols)? {
            xs.extend(col.f32()?.into_no_null_iter());
        }
        let xs = Tensor::from_vec(xs, (node_df.height(), self.feature_cols.len()), device)?;

        let edge_df = self
            .edge_df
            .inner_join(&index, ["source"], ["id"])?
            .inner_join(&index, ["target"], ["id"])?;
        let mut edge_index = Vec::new();
        edge_index.extend(edge_df["__index"].u32()?.into_no_null_iter());
        edge_index.extend(edge_df["__index_right"].u32()?.into_no_null_iter());
        let edge_index = Tensor::from_vec(edge_index, (2, edge_df.height()), device)?;

        let masked_node_df = node_df.filter(node_df["mask"].bool()?)?;
        let ys = Tensor::from_iter(
            masked_node_df["label_u32"].u32()?.into_no_null_iter(),
            device,
        )?;
        let mask = Tensor::from_iter(masked_node_df["__index"].u32()?.into_no_null_iter(), device)?;

        Ok(CiteSeerBatch {
            xs,
            edge_index,
            ys,
            mask,
        })
    }
}
