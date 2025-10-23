from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

class GraphDataset(Dataset):
    
    def len(self):
        return ("xx")
    def get(self, idx):
        return ("xx")
    
    
    
    
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):

        graph = self.graphs[idx]
        # return {'node_features': graph.node_features, 'edge_index': graph.edge_index}
        return graph