from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args
from prompt_graph.data import load4node,load4graph, split_induced_graphs
import pickle
import random
import numpy as np
import os
import pandas as pd

import re
def load_induced_graph(dataset_name, data, device):

    folder_path = './Experiment/induced_graph/' + dataset_name
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    if dataset_name in ['Wisconsin','Texas','chameleon']:
        file_path = folder_path + '/induced_graph_min1_max300.pkl'
    elif dataset_name in ['WikiCS','Photo','Computers']:
        file_path = folder_path + '/induced_graph_min1_max300.pkl'
    elif dataset_name in ['CS','Physics','Reddit']:
        file_path = folder_path + '/induced_graph_min10_max30.pkl'
    elif dataset_name in ['squirrel','Cornell']:
        file_path = folder_path + '/induced_graph_min1_max300.pkl'
    else:
        file_path = folder_path + '/induced_graph_min100_max300.pkl'
    if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                print('loading induced graph...')
                graphs_list = pickle.load(f)
                print('Done!!!')
    else:
        print('Begin split_induced_graphs.')
        if dataset_name in ['Wisconsin','Texas','chameleon']:
            split_induced_graphs(dataset_name,data, folder_path, device, smallest_size=1, largest_size=300)
        elif dataset_name in ['WikiCS','Photo','Computers']:
            split_induced_graphs(dataset_name,data, folder_path, device, smallest_size=1, largest_size=300)
        elif dataset_name in ['CS','Physics','Reddit']:
            split_induced_graphs(dataset_name,data, folder_path, device, smallest_size=10, largest_size=30)
        elif dataset_name in ['squirrel','Cornell']:
            split_induced_graphs(dataset_name,data, folder_path, device, smallest_size=1, largest_size=300)
        else:
            split_induced_graphs(dataset_name,data, folder_path, device, smallest_size=100, largest_size=300)
        with open(file_path, 'rb') as f:
            graphs_list = pickle.load(f)
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list




args = get_args()
seed_everything(args.seed)

if args.pre_train_model_path=='None':
    pre = 'None'
else:
    pre = re.search(r'^[^/]+/[^/]+/[^/]+/[^/]+/([^\.]+)\.[^\.]+', args.pre_train_model_path).group(1)




print('dataset_name', args.dataset_name)


if args.task == 'NodeTask':
    data, input_dim, output_dim = load4node(args.dataset_name)   
    data = data.to(args.device)
    if args.prompt_type in ['Gprompt', 'All-in-one', 'GSFP', 'GSmFP','None']:
        graphs_list = load_induced_graph(args.dataset_name, data, args.device)
    else:
        graphs_list = None 
            
         

if args.task == 'GraphTask':
    input_dim, output_dim, dataset = load4graph(args.dataset_name)

if args.task == 'NodeTask':
    tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer,
                    gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type,
                    epochs = args.epochs, shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                    batch_size = args.batch_size, data = data, input_dim = input_dim, output_dim = output_dim, graphs_list = graphs_list,lambda_=args.lambda_,pnum=args.pnum)

if args.task == 'GraphTask':
    tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = args.epochs,
                    shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                    batch_size = args.batch_size, dataset = dataset, input_dim = input_dim, output_dim = output_dim,lambda_=args.lambda_,pnum=args.pnum)
pre_train_type = tasker.pre_train_type


_, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _,val_acc,std_val_acc= tasker.run()

print("End! Final Val Accuracy", val_acc)
print("End! Final Val Std", std_val_acc)
print("End! Final Test Accuracy {:.4f}".format(test_acc)) 
print("End! Final Test Std {:.4f}".format(std_test_acc)) 



