import torch
from torch_geometric.loader import DataLoader
from prompt_graph.utils import constraint,  center_embedding, Gprompt_tuning_loss
from prompt_graph.evaluation import GPPTEva, GSFPEva,GNNNodeEva

from .task import BaseTask
import time
import warnings
import numpy as np
from prompt_graph.data import load4node, induced_graphs, graph_split, split_induced_graphs, node_sample_and_save,GraphDataset
from prompt_graph.evaluation import GpromptEva, AllInOneEva
import pickle
import os
from prompt_graph.utils import process
import torchmetrics

warnings.filterwarnings("ignore")

class NodeTask(BaseTask):
      def __init__(self, data, input_dim, output_dim, graphs_list = None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.task_type = 'NodeTask'
            if self.prompt_type == 'MultiGprompt':
                  self.load_multigprompt_data()
            else:
                  self.data = data
                  if self.dataset_name == 'ogbn-arxiv':
                        self.data.y = self.data.y.squeeze()
                  self.input_dim = input_dim
                  self.output_dim = output_dim
                  self.graphs_list = graphs_list            
            self.create_few_data_folder()

      def create_few_data_folder(self):
            for k in range(1, 11):
                  k_shot_folder = './Experiment/sample_data/Node/'+ self.dataset_name +'/' + str(k) +'_shot'
                  os.makedirs(k_shot_folder, exist_ok=True)
                  
                  for i in range(1, 6):
                        folder = os.path.join(k_shot_folder, str(i))
                        if not os.path.exists(folder):
                              os.makedirs(folder)
                              node_sample_and_save(self.data, k, folder, self.output_dim)
                              print(str(k) + ' shot ' + str(i) + ' th is saved!!')

      def load_multigprompt_data(self):
            adj, features, labels = process.load_data(self.dataset_name)
            self.input_dim = features.shape[1]
            self.output_dim = labels.shape[1]
            print('a',self.output_dim)
            features, _ = process.preprocess_features(features)
            self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
            self.labels = torch.FloatTensor(labels[np.newaxis])
            self.features = torch.FloatTensor(features[np.newaxis]).to(self.device)
            print("adj",self.sp_adj.shape)
            print("feature",features.shape)

      
      def load_data(self):
            self.data, self.input_dim, self.output_dim = load4node(self.dataset_name)


      def Train(self, train_loader):
            self.gnn.train()
            self.answering.train()
            self.embedding.train()
            
            total_loss = 0.0 
            for batch in train_loader:  
                  self.optimizer.zero_grad() 
                  batch = batch.to(self.device)
                  
                  batch.x = self.embedding(batch.x)
                  
                  out = self.gnn(batch.x, batch.edge_index, batch.batch)
                  out = self.answering(out)
                  loss = self.criterion(out, batch.y)  
                  loss.backward()  
                  self.optimizer.step()                                  
                  total_loss += loss.item()  
            
            return total_loss / len(train_loader)   
      
      
          
      
      def GPPTtrain(self, data, train_idx):
            
            self.gnn.eval()
            self.embedding.train()
            self.prompt.train()
            
            embedding_x = self.embedding(data.x)
            
            node_embedding = self.gnn(embedding_x, data.edge_index)
            out = self.prompt(node_embedding, data.edge_index)
            loss = self.criterion(out[train_idx], data.y[train_idx])
            loss = loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
            self.pg_opi.zero_grad()
            loss.backward()
            self.pg_opi.step()
            mid_h = self.prompt.get_mid_h()
            self.prompt.update_StructureToken_weight(mid_h)
            return loss.item()
      
      
            
      def GSFPTrain(self, train_loader):
            self.gnn.eval()
            self.answering.train()
            self.embedding.train()
            self.prompt.train()
            total_loss = 0.0 
            for batch in train_loader:  
                  self.optimizer.zero_grad() 
                  batch = batch.to(self.device)
                  
                  embedding_x = self.embedding(batch.x)                 
                  prompt_x = self.prompt.add(embedding_x)
                  
                  out = self.gnn(prompt_x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = self.prompt_type)
                  out = self.answering(out)
                  loss = self.criterion(out, batch.y)  
                  loss.backward()  
                  self.optimizer.step()

                  if self.lambda_ != 0.:
                        self.optimizer_l1_l21.zero_grad()
                        self.optimizer_l1_l21.step() 

                  total_loss += loss.item()  
            
            return total_loss / len(train_loader) 
      
      
      

      def AllInOneTrain(self, train_loader, answer_epoch=1, prompt_epoch=1):
    
            self.gnn.eval()
            self.answering.train()
            self.embedding.train()
            self.prompt.eval()
            
            for epoch in range(1, answer_epoch + 1):
                  answer_loss = self.prompt.Tune(self.embedding,train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
                  print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, answer_loss)))

            # tune prompt
            self.gnn.eval()
            self.answering.eval()
            self.embedding.eval()
            self.prompt.train()
            
            for epoch in range(1, prompt_epoch + 1):
                  pg_loss = self.prompt.Tune( self.embedding,train_loader,  self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
                  print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, prompt_epoch, pg_loss)))
            
            # return pg_loss
            return answer_loss
      
      def GpromptTrain(self, train_loader):
            
            self.gnn.eval()
            self.embedding.train()
            self.prompt.train()
            
            total_loss = 0.0 
            accumulated_centers = None
            accumulated_counts = None
            for batch in train_loader:  
                  self.pg_opi.zero_grad() 
                  batch = batch.to(self.device)
                  
                  embedding_x = self.embedding(batch.x) 
                  
                  out = self.gnn(embedding_x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = 'Gprompt')
                  # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
                  center, class_counts = center_embedding(out, batch.y, self.output_dim)
        
                  if accumulated_centers is None:
                        accumulated_centers = center
                        accumulated_counts = class_counts
                  else:
                        accumulated_centers += center * class_counts
                        accumulated_counts += class_counts
                  criterion = Gprompt_tuning_loss()
                  loss = criterion(out, center, batch.y)  
                  loss.backward()  
                  self.pg_opi.step()  
                  total_loss += loss.item()
    
            mean_centers = accumulated_centers / accumulated_counts

            return total_loss / len(train_loader), mean_centers
      
      def run(self):
            test_accs = []
            
            val_accs = []
            f1s = []
            rocs = []
            prcs = []
            batch_best_loss = []
            if self.prompt_type == 'All-in-one':
                  self.answer_epoch = 50
                  self.prompt_epoch = 50
                  self.epochs = int(self.epochs/self.answer_epoch)
            for i in range(1, 6):     
                  torch.cuda.empty_cache()
                  self.initialize_gnn()
                  self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                                                  torch.nn.Softmax(dim=1)).to(self.device)
                  self.embedding =  torch.nn.Sequential(torch.nn.Linear(self.input_dim, 100, bias = False)).to(self.device)
                  self.initialize_prompt()
                  self.initialize_optimizer()
                  idx_train = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                  train_lbls = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
                  idx_test = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                  test_lbls = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
                  idx_val = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/val_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                  val_lbls = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/val_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)

                  
                  # GPPT prompt initialtion
                  if self.prompt_type == 'GPPT':
                        
                        embedding_x = self.embedding(self.data.x)
                        
                        node_embedding = self.gnn(embedding_x, self.data.edge_index)
                        self.prompt.weigth_init(node_embedding,self.data.edge_index, self.data.y, idx_train)

                  
                  if self.prompt_type in ['Gprompt', 'All-in-one', 'GSFP', 'GSmFP','None']:
                  
                        train_graphs = []
                        test_graphs = []
                        val_graphs = []
                        
                        print('distinguishing the train dataset and test dataset...')
                        for graph in self.graphs_list:                              
                              if graph.index in idx_train:
                                    train_graphs.append(graph)
                              elif graph.index in idx_test:
                                    test_graphs.append(graph)
                              elif graph.index in idx_val:
                                    val_graphs.append(graph)
                        print('Done!!!')

                        train_dataset = GraphDataset(train_graphs)
                        test_dataset = GraphDataset(test_graphs)
                        val_dataset = GraphDataset(val_graphs)

                        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                        print("prepare induce graph data is finished!")



                  patience = 20
                  best = 1e9
                  cnt_wait = 0
                  best_loss = 1e9
                  best_epoch =-1

                  
                  for epoch in range(1, self.epochs):
                        t0 = time.time()
                        if self.prompt_type == 'None':
                              loss = self.Train(train_loader)                          
                        elif self.prompt_type == 'GPPT':
                              loss = self.GPPTtrain(self.data, idx_train)                
                        elif self.prompt_type == 'All-in-one':
                              loss = self.AllInOneTrain(train_loader,self.answer_epoch,self.prompt_epoch)                           
                        elif self.prompt_type in ['GSFP', 'GSmFP']:
                              loss = self.GSFPTrain(train_loader)                                                       
                        elif self.prompt_type =='Gprompt':
                              loss, center = self.GpromptTrain(train_loader)

        
                         
                        if loss < best:
                              best_epoch = epoch
                              best = loss                                               
                              cnt_wait = 0
                                                                                                                        
                        else:
                              cnt_wait += 1
                              if cnt_wait == patience:
                                    print('-' * 100)
                                    print('Early stopping at '+str(epoch) +' eopch!')
                                    break
                                             
                        print("Epoch {:03d} |  Time(s) {:.4f} |Train Loss {:.4f}".format(epoch, time.time() - t0, loss))


                  import math
                  if not math.isnan(loss):
                        batch_best_loss.append(loss)
                  
                        if self.prompt_type == 'None':
                              test_acc, f1, roc, prc = GNNNodeEva(test_loader,self.gnn, self.answering,self.embedding, self.output_dim, self.device)    
                              val_acc, val_f1,val_roc, val_prc = GNNNodeEva(val_loader,self.gnn, self.answering, self.embedding,self.output_dim, self.device)                                                                                                  
                        elif self.prompt_type == 'GPPT':
                              test_acc, f1, roc, prc = GPPTEva(self.data, idx_test,self.embedding, self.gnn, self.prompt, self.output_dim, self.device)   
                              val_acc, val_f1,val_roc, val_prc  = GPPTEva(self.data, idx_val, self.embedding,self.gnn, self.prompt, self.output_dim, self.device)                          
                        elif self.prompt_type == 'All-in-one':
                              test_acc, f1, roc, prc = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering,self.embedding, self.output_dim, self.device)    
                              val_acc, val_f1,val_roc, val_prc = AllInOneEva(val_loader, self.prompt, self.gnn, self.answering,self.embedding, self.output_dim, self.device)                                         
                        elif self.prompt_type in ['GSFP', 'GSmFP']:
                              test_acc, f1, roc, prc = GSFPEva(test_loader, self.gnn, self.prompt, self.answering,self.embedding, self.output_dim, self.device)
                              val_acc, val_f1,val_roc, val_prc = GSFPEva(val_loader, self.gnn, self.prompt, self.answering,self.embedding, self.output_dim, self.device)                                                                                     

                        elif self.prompt_type =='Gprompt':
                              test_acc, f1, roc, prc = GpromptEva(test_loader, self.embedding,self.gnn, self.prompt, center, self.output_dim, self.device)
                              val_acc, val_f1,val_roc, val_prc = GpromptEva(val_loader, self.embedding,self.gnn, self.prompt, center, self.output_dim, self.device)

                        print(f"One task completed. Val Accuracy: {val_acc:.4f}")                       
                        print(f"One task completed. Test Accuracy: {test_acc:.4f}")

                               
                        test_accs.append(test_acc)
                        f1s.append(f1)
                        rocs.append(roc)
                        prcs.append(prc)
                        
                        val_accs.append(val_acc)
        
            mean_test_acc = np.mean(test_accs)
            std_test_acc = np.std(test_accs)    
            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)   
            mean_roc = np.mean(rocs)
            std_roc = np.std(rocs)   
            mean_prc = np.mean(prcs)
            std_prc = np.std(prcs) 
            
            mean_val_acc = np.mean(val_accs)
            std_val_acc = np.std(val_accs)              
            
             
            print(self.pre_train_type, self.gnn_type, self.prompt_type, " Node Task completed")
            mean_best = np.mean(batch_best_loss)

            return  mean_best, mean_test_acc, std_test_acc, mean_f1, std_f1, mean_roc, std_roc, mean_prc, std_prc,mean_val_acc,std_val_acc

