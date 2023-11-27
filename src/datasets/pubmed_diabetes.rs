use std::{
    collections::HashMap,
    fs::{create_dir_all, File},
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use polars::{
    chunked_array::ops::ChunkFull,
    datatypes::BooleanChunked,
    io::{
        parquet::{ParquetReader, ParquetWriter},
        SerReader,
    },
    prelude::{df, DataFrame, DataFrameJoinOps, NamedFrom, NamedFromOwned, Series},
};
use regex::Regex;

use super::{download_and_extract, PolarsDataset};
use super::{traits::Dataset, CompressionFormat};

#[derive(Debug, Clone)]
pub struct PubMedDiabetesBatch {
    pub xs: Tensor,
    pub edge_index: Tensor,
    pub ys: Tensor,
    pub mask: Tensor, // loss(&logits.i(mask)?, &ys)
}

#[derive(Debug, Clone)]
pub struct PubMedDiabetesDataset {
    node_df: DataFrame,
    edge_df: DataFrame,
}
impl PubMedDiabetesDataset {
    const NUM_FEATURES: usize = 500;
    const NUM_CLASSES: usize = 3;
    const NUM_NODES: usize = 19717;
    const NUM_EDGES: usize = 44338;

    pub fn prepare_data<P: AsRef<Path>>(root: P) -> anyhow::Result<()> {
        let raw = root.as_ref().join("raw");
        if !raw.exists() {
            create_dir_all(&raw)?;
            download_and_extract(
                "https://linqs-data.soe.ucsc.edu/public/datasets/pubmed-diabetes/pubmed-diabetes.tar.gz",
                &raw,
                CompressionFormat::Tgz,
            )?;
        }
        let processed = root.as_ref().join("processed");
        if !processed.exists() {
            create_dir_all(&processed)?;
            let e = || anyhow::anyhow!("Exhausted Iterator");
            {
                let reader = BufReader::new(File::open(
                    raw.join("pubmed-diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab"),
                )?);
                let mut source = Vec::new();
                let mut target = Vec::new();
                let mut lines_iter = reader.lines();
                let _ = lines_iter.next().ok_or(anyhow!("failed to read header 1")); // header 1
                let _ = lines_iter.next().ok_or(anyhow!("failed to read header 2")); // header 2

                let regex = Regex::new(r"\d+\s+paper:(\d*)\s*\|\s*paper:(\d*)").unwrap();
                for buf in lines_iter {
                    let line = buf.unwrap();
                    if let Some(c) = regex.captures(&line) {
                        let u = c
                            .get(1)
                            .ok_or(anyhow!(format!("failed to parse u; {:?}", c)))?
                            .as_str()
                            .parse::<u32>()?;
                        let v = c
                            .get(2)
                            .ok_or(anyhow!(format!("failed to parse v; {:?}", c)))?
                            .as_str()
                            .parse::<u32>()?;
                        source.push(u);
                        target.push(v);
                    } else {
                        return Err(anyhow::anyhow!(format!("don't capture; {:?}", line)));
                    }
                }
                assert_eq!(source.len(), Self::NUM_EDGES);
                let mut edge_df = df! {
                    "source" => source,
                    "target" => target,
                }?;
                ParquetWriter::new(File::create(processed.join("edges.parquet"))?)
                    .finish(&mut edge_df)?;
            }
            {
                let reader = BufReader::new(File::open(
                    raw.join("pubmed-diabetes/data/Pubmed-Diabetes.NODE.paper.tab"),
                )?);
                let mut id = Vec::new();
                let mut xs = vec![vec![0.0; Self::NUM_NODES]; Self::NUM_FEATURES];
                let mut label = Vec::new();

                let mut lines_iter = reader.lines();
                let _ = lines_iter.next().ok_or_else(e)?; // header 1

                let mut feature_idx = HashMap::new();
                let header = lines_iter.next().ok_or_else(e)??; // header 2
                let mut entries = header.split_whitespace();
                assert_eq!(entries.next(), Some("cat=1,2,3:label")); // discard first element
                let regex = Regex::new(r"numeric:(.*):(.*)").unwrap();

                for (idx, entry) in entries.enumerate() {
                    if let Some(c) = regex.captures(entry) {
                        let key = c.get(1).ok_or_else(e)?.as_str().to_owned();
                        assert_eq!(c.get(2).ok_or_else(e)?.as_str(), "0.0");
                        feature_idx.insert(key, idx);
                    }
                }
                assert_eq!(feature_idx.len(), Self::NUM_FEATURES);

                let regex = Regex::new(r"([^\s=]+)=([\d|.]+)").unwrap();
                for (rank, buf) in lines_iter.enumerate() {
                    let line = buf?;
                    let mut entries = line.split_whitespace();
                    id.push(entries.next().ok_or_else(e)?.parse::<u32>()?);
                    let entry = entries.next().ok_or_else(e)?;
                    if let Some(c) = regex.captures(entry) {
                        assert_eq!(c.get(1).map(|m| m.as_str()), Some("label"));
                        label.push(
                            c.get(2)
                                .ok_or(anyhow!(format!("failed to parse {:?}", c)))?
                                .as_str()
                                .to_owned(),
                        );
                    }
                    for entry in entries {
                        if let Some(c) = regex.captures(entry) {
                            let key = c
                                .get(1)
                                .ok_or(anyhow!(format!("failed to parse {:?}", c)))?
                                .as_str();
                            let val = c
                                .get(2)
                                .ok_or(anyhow!(format!("failed to parse {:?}", c)))?
                                .as_str()
                                .parse::<f32>()?;
                            xs[feature_idx[key]][rank] = val;
                        }
                    }
                }
                assert_eq!(id.len(), Self::NUM_NODES);
                let mut node_df = df! {
                    "id" => id,
                    "label" => label,
                }?;
                for (i, x) in xs.into_iter().enumerate() {
                    let name = format!("xs.{}", i);
                    node_df.with_column(Series::from_vec(&name, x))?;
                }
                ParquetWriter::new(File::create(processed.join("nodes.parquet"))?)
                    .finish(&mut node_df)?;
            }
        }
        Ok(())
    }

    pub fn from_processed<P: AsRef<Path>>(root: P) -> anyhow::Result<Self> {
        let path = root.as_ref().join("processed");
        let mut node_df = ParquetReader::new(File::open(path.join("nodes.parquet"))?).finish()?;
        let mut edge_df = ParquetReader::new(File::open(path.join("edges.parquet"))?).finish()?;

        // assign u32 label
        let label = DataFrame::new(vec![node_df["label"].unique_stable()?])?
            .with_row_count("label_u32", None)?;
        node_df = node_df.inner_join(&label, ["label"], ["label"])?;

        // assign mask
        node_df.with_column(BooleanChunked::full("mask", true, node_df.height()))?;

        // make undirectional
        let mut rev_edge_df = edge_df.clone();
        rev_edge_df.replace("source", edge_df["target"].clone())?;
        rev_edge_df.replace("target", edge_df["source"].clone())?;
        edge_df = edge_df.vstack(&rev_edge_df)?;
        Ok(Self { node_df, edge_df })
    }

    pub fn new<P: AsRef<Path>>(root: P) -> anyhow::Result<Self> {
        let root = root.as_ref();
        Self::prepare_data(root)?;
        Self::from_processed(root)
    }
    pub fn feature_cols(&self) -> Vec<String> {
        (0..self.num_features())
            .map(|i| format!("xs.{}", i))
            .collect()
    }
    pub fn num_features(&self) -> usize {
        Self::NUM_FEATURES
    }
    pub fn num_classes(&self) -> usize {
        Self::NUM_CLASSES
    }
    fn id_cols(&self) -> &[&str] {
        &["id"]
    }
}

impl PolarsDataset for PubMedDiabetesDataset {
    fn node_df(&self) -> &DataFrame {
        &self.node_df
    }
    fn edge_df(&self) -> &DataFrame {
        &self.edge_df
    }
    fn with_node_df(&self, node_df: DataFrame) -> Self {
        Self {
            node_df,
            edge_df: self.edge_df.clone(),
        }
    }
    fn with_edge_df(&self, edge_df: DataFrame) -> Self {
        Self {
            node_df: self.node_df.clone(),
            edge_df,
        }
    }
}

impl Dataset for PubMedDiabetesDataset {
    type Batch = PubMedDiabetesBatch;
    type NodeSelector = DataFrame;

    fn all_nodes(&self) -> Result<DataFrame> {
        let result = self.node_df.select(self.id_cols())?;
        Ok(result)
    }
    fn induced_subgraph(&self, nodes: DataFrame, device: &Device) -> Result<Self::Batch> {
        let index = nodes.with_row_count("__index", None)?;

        let node_df = index.inner_join(&self.node_df, ["id"], ["id"])?;
        let mut xs = Vec::new();
        for col in node_df.select_series(self.feature_cols())? {
            xs.extend(col.f32()?.into_no_null_iter());
        }
        let xs = Tensor::from_vec(xs, (node_df.height(), self.num_features()), device)?;

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

        Ok(Self::Batch {
            xs,
            edge_index,
            ys,
            mask,
        })
    }
}
