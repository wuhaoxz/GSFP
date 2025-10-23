import torch
import pickle as pk
from random import shuffle
import random
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr, WebKB, Actor, Coauthor
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data,Batch
from torch_geometric.utils import negative_sampling
import os
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset


def node_sample_and_save(data, k, folder, num_classes):
    labels = data.y.to('cpu')
    perm = torch.randperm(data.num_nodes)
    sampled_train_idx = []
    
    for i in range(num_classes):
        class_idx = perm[labels[perm] == i]
        if len(class_idx) < k:
            sampled_train_idx.append(class_idx)
        else:
            sampled_train_idx.append(class_idx[:k])
    
    sampled_train_idx = torch.cat(sampled_train_idx)

    shuffled_indices = torch.randperm(sampled_train_idx.size(0))
    sampled_train_idx = sampled_train_idx[shuffled_indices]
    
    remaining_idx = perm[~torch.isin(perm, sampled_train_idx)]

    num_remaining = remaining_idx.size(0)
    num_val = int(0.1 * num_remaining)
    num_test = num_remaining - num_val 

    val_idx = remaining_idx[:num_val]
    test_idx = remaining_idx[num_val:num_val + num_test]


    train_labels = labels[sampled_train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    
    torch.save(sampled_train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(val_idx, os.path.join(folder, 'val_idx.pt'))
    torch.save(val_labels, os.path.join(folder, 'val_labels.pt'))
    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))
    


def graph_sample_and_save(dataset, k, folder, num_classes):

    num_graphs = len(dataset)
    labels = torch.tensor([graph.y.item() for graph in dataset])
    all_indices = torch.randperm(num_graphs)
    
    train_indices = []

    for i in range(num_classes):
        class_indices = [idx for idx in all_indices if labels[idx].item() == i]
        if len(class_indices) < k:
            selected_indices = class_indices
        else:
            selected_indices = class_indices[:k]
        train_indices.extend(selected_indices)
    
    train_indices = torch.tensor(train_indices)
    shuffled_train_indices = torch.randperm(train_indices.size(0))
    train_indices = train_indices[shuffled_train_indices]
    remaining_indices = all_indices[~torch.isin(all_indices, train_indices)]

    num_remaining = remaining_indices.size(0)
    num_val = int(0.1 * num_remaining)
    num_test = num_remaining - num_val
    
    val_indices = remaining_indices[:num_val]
    test_indices = remaining_indices[num_val:num_val + num_test]

    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]
    
    torch.save(train_indices, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(val_indices, os.path.join(folder, 'val_idx.pt'))
    torch.save(val_labels, os.path.join(folder, 'val_labels.pt'))
    torch.save(test_indices, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))




def node_degree_as_features(data_list):
    from torch_geometric.utils import degree
    for data in data_list:
        deg = degree(data.edge_index[0], dtype=torch.long)
        deg = deg.view(-1, 1).float()
        if data.x is None:
            data.x = deg
        else:
            data.x = torch.cat([data.x, deg], dim=1)


def load4graph(dataset_name, shot_num= 10, num_parts=None, pretrained=False):
    r"""A plain old python object modeling a batch of graphs as one big
        (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
        base class, all its methods can also be used here.
        In addition, single graphs can be reconstructed via the assignment vector
        :obj:`batch`, which maps each node to its respective graph identifier.
        """

    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'DD','IMDB-MULTI','PTC_FM','PTC_FR','PTC_MM','NCI1','NCI109','REDDIT-MULTI-5K']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)
        input_dim = dataset.num_features
        out_dim = dataset.num_classes


        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        if dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY','IMDB-MULTI','REDDIT-MULTI-5K']:
            graph_list = [g for g in graph_list]
            node_degree_as_features(graph_list)
            input_dim = graph_list[0].x.size(1)   
   
        if(pretrained==True):
            return input_dim, out_dim, graph_list
        else:
            return input_dim, out_dim, dataset
        
    if dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = PygGraphPropPredDataset(name = dataset_name, root='./dataset')
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        graph_list = [g for g in graph_list]
        node_degree_as_features(graph_list)
        input_dim = graph_list[0].x.size(1)

        for g in graph_list:
            g.y = g.y.squeeze(0)

        if(pretrained==True):
            return input_dim, out_dim, graph_list
        else:
            return input_dim, out_dim, dataset        

    
def load4node(dataname):
    print(dataname)
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataname, transform=NormalizeFeatures())
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='data/amazon', name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
        
    elif dataname in ['Physics', 'CS']:
        dataset = Coauthor(root='data/Coauthor', name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Flickr':
        dataset = Flickr(root='data/Flickr')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['Wisconsin', 'Texas','Cornell']:
        dataset = WebKB(root='data/'+dataname, name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root='data/'+dataname, name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes   
    elif dataname == 'Actor':
        dataset = Actor(root='data/Actor')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset')
        data = dataset[0]
        input_dim = data.x.shape[1]
        out_dim = dataset.num_classes

    return data, input_dim, out_dim


def load4link_prediction_single_graph(dataname, num_per_samples=1):
    data, input_dim, output_dim = load4node(dataname)

    
    r"""Perform negative sampling to generate negative neighbor samples"""
    if data.is_directed():
        row, col = data.edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = data.edge_index
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges * num_per_samples,
    )

    edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)

    return data, edge_label, edge_index, input_dim, output_dim

def load4link_prediction_multi_graph(dataset_name, num_per_samples=1):
    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'DD']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)

    if dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = PygGraphPropPredDataset(name = dataset_name, root='./dataset')
    
    input_dim = dataset.num_features
    output_dim = 2

    if dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']:
        dataset = [g for g in dataset]
        node_degree_as_features(dataset)
        input_dim = dataset[0].x.size(1)

    elif dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = [g for g in dataset]
        node_degree_as_features(dataset)
        input_dim = dataset[0].x.size(1)
        for g in dataset:
            g.y = g.y.squeeze(1)

    data = Batch.from_data_list(dataset)
    
    r"""Perform negative sampling to generate negative neighbor samples"""
    if data.is_directed():
        row, col = data.edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = data.edge_index
        
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges * num_per_samples,
    )

    edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)
    
    return data, edge_label, edge_index, input_dim, output_dim

def load4link_prediction_multi_large_scale_graph(dataset_name, num_per_samples=1):
    if dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = PygGraphPropPredDataset(name = dataset_name, root='./dataset')
    
    input_dim = dataset.num_features
    output_dim = 2

    dataset = [g for g in dataset]
    node_degree_as_features(dataset)
    input_dim = dataset[0].x.size(1)
    for g in dataset:
        g.y = g.y.squeeze(1)

    batch_graph_num = 20000
    split_num = int(len(dataset)/batch_graph_num)
    data_list = []
    edge_label_list = []
    edge_index_list = []
    for i in range(split_num+1):
        if(i==0):
            data = Batch.from_data_list(dataset[0:batch_graph_num])
        elif(i<=split_num):
            data = Batch.from_data_list(dataset[i*batch_graph_num:(i+1)*batch_graph_num])
        elif len(dataset)>((i-1)*batch_graph_num):
            data = Batch.from_data_list(dataset[i*batch_graph_num:(i+1)*batch_graph_num])
        

        data_list.append(data)
        
        r"""Perform negative sampling to generate negative neighbor samples"""
        if data.is_directed():
            row, col = data.edge_index
            row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
            edge_index = torch.stack([row, col], dim=0)
        else:
            edge_index = data.edge_index
            
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.num_edges * num_per_samples,
        )

        edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)
    
    return data, edge_label, edge_index, input_dim, output_dim


def NodePretrain(dataname='CiteSeer', num_parts=200):
    r"""Load different datasets, 
    get the number of node features and divide data into multiple clusters."""
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataname, transform=NormalizeFeatures())
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='data/amazon', name=dataname)
    elif dataname in ['Physics', 'CS']:
        dataset = Coauthor(root='data/Coauthor', name=dataname)
    elif dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit')
    elif dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS')
    elif dataname == 'Flickr':
        dataset = Flickr(root='data/Flickr')
    elif dataname in ['Wisconsin', 'Texas','Cornell']:
        dataset = WebKB(root='data/'+dataname, name=dataname)
    elif dataname in ['chameleon', 'squirrel']:    
        dataset = WikipediaNetwork(root='data/'+dataname, name=dataname)     
    elif dataname == 'Actor':
        dataset = Actor(root='data/Actor')
    elif dataname == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset')


    data = dataset[0]

    x = data.x.detach()
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    data = Data(x=x, edge_index=edge_index)
    input_dim = data.x.shape[1]
    graph_list = list(ClusterData(data=data, num_parts=num_parts))

    return graph_list, input_dim


