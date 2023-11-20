// dataset = torch_geometric.datasets.MovieLens("./")
struct HeteroData {
    x: HashMap<String, Tensor>,
    edge_index: HashMap<(String, String, String), Tensor>,
    edge_label: HashMap<(String, String, String), Tensor>,
    time: HashMap<(String, String, String), Tensor>,
}

