import torch
from prompt_graph.data import load4graph, load4node, graph_sample_and_save
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from .task import BaseTask
from prompt_graph.utils import center_embedding, Gprompt_tuning_loss,constraint
from prompt_graph.evaluation import GpromptEva, GNNGraphEva, GSFPEva, AllInOneEva, GPPTGraphEva
import time
import os 
import numpy as np

class GraphTask(BaseTask):
    def __init__(self, input_dim, output_dim, dataset, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.task_type = 'GraphTask'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dataset = dataset
        if self.shot_num > 0:
            self.create_few_data_folder()


    def create_few_data_folder(self):
       
        for k in range(1, 11):
                k_shot_folder = './Experiment/sample_data/Graph/'+ self.dataset_name +'/' + str(k) +'_shot'
                os.makedirs(k_shot_folder, exist_ok=True)
                
                for i in range(1, 6):
                    folder = os.path.join(k_shot_folder, str(i))
                    if not os.path.exists(folder):
                        os.makedirs(folder, exist_ok=True)
                        graph_sample_and_save(self.dataset, k, folder, self.output_dim)
                        print(str(k) + ' shot ' + str(i) + ' th is saved!!')

    def load_data(self):
        if self.dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'ogbg-ppa','DD','IMDB-MULTI','REDDIT-MULTI-5K']:
            self.input_dim, self.output_dim, self.dataset= load4graph(self.dataset_name, self.shot_num)

    def node_degree_as_features(self, data_list):
        from torch_geometric.utils import degree
        for data in data_list:
            deg = degree(data.edge_index[0], dtype=torch.long)
            deg = deg.view(-1, 1).float()
            if data.x is None:
                data.x = deg
            else:
                data.x = torch.cat([data.x, deg], dim=1)

    def Train(self, train_loader):

        self.gnn.train()
        self.answering.train()
        self.embedding.train()

        total_loss = 0.0 
        for batch in train_loader:  
            self.optimizer.zero_grad() 
            batch = batch.to(self.device)
            
            embedding_x = self.embedding(batch.x)
            
            out = self.gnn(embedding_x, batch.edge_index, batch.batch)
            out = self.answering(out)
            loss = self.criterion(out, batch.y)  

            
            loss.backward()  
            self.optimizer.step()  
            total_loss += loss.item()  
        return total_loss / len(train_loader)  
        
    def AllInOneTrain(self, train_loader, answer_epoch=1, prompt_epoch=1):
        #we update answering and prompt alternately.

        # tune task head
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
            print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, pg_loss)))
        
        return pg_loss

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
            center, class_counts = center_embedding(out,batch.y, self.output_dim)

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

    def GPPTtrain(self, train_loader):
        self.gnn.eval()
        self.embedding.train()
        self.prompt.train()
        for batch in train_loader:
            temp_loss=torch.tensor(0.0,requires_grad=True).to(self.device)
            graph_list = batch.to_data_list()        
            for index, graph in enumerate(graph_list):
                graph=graph.to(self.device)  
                
                graph.x = self.embedding(graph.x)
                            
                node_embedding = self.gnn(graph.x,graph.edge_index)
                out = self.prompt(node_embedding, graph.edge_index)
                loss = self.criterion(out, torch.full((1,graph.x.shape[0]), graph.y.item()).reshape(-1).to(self.device))
                temp_loss += loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())           
            temp_loss = temp_loss/(index+1)
            self.pg_opi.zero_grad()
            temp_loss.backward()
            self.pg_opi.step()
            self.prompt.update_StructureToken_weight(self.prompt.get_mid_h())
        return temp_loss.item()

    def run(self):
        test_accs = []

        val_accs = []

        f1s = []
        rocs = []
        prcs = []
        batch_best_loss = []
        if self.prompt_type == 'All-in-one':
            # self.answer_epoch = 5 MUTAG Graph MAE / GraphCL
            # self.prompt_epoch = 1
            self.answer_epoch = 50
            self.prompt_epoch = 50
            self.epochs = int(self.epochs/self.answer_epoch)
        if self.shot_num > 0:
            for i in range(1, 6):
                torch.cuda.empty_cache()
                self.initialize_gnn()
                self.initialize_prompt()
                self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                                    torch.nn.Softmax(dim=1)).to(self.device)
                self.embedding =  torch.nn.Sequential(torch.nn.Linear(self.input_dim, 100, bias = False)).to(self.device)

                self.initialize_optimizer()


                idx_train = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                train_lbls = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
                idx_test = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                test_lbls = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
                idx_val = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/val_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                val_lbls = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/val_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)



                train_dataset = self.dataset[idx_train]
                test_dataset = self.dataset[idx_test]
     
                val_dataset = self.dataset[idx_val]

                if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa','IMDB-MULTI','REDDIT-MULTI-5K']:
                    from torch_geometric.data import Batch
                    train_dataset = [train_g for train_g in train_dataset]
                    test_dataset = [test_g for test_g in test_dataset]
                    val_dataset = [val_g for val_g in val_dataset]
                    self.node_degree_as_features(train_dataset)
                    self.node_degree_as_features(test_dataset)
                    self.node_degree_as_features(val_dataset)
         
                    if self.prompt_type == 'GPPT':
                        processed_dataset = [g for g in self.dataset]
                        self.node_degree_as_features(processed_dataset)
                        processed_dataset = Batch.from_data_list([g for g in processed_dataset])
                    self.input_dim = train_dataset[0].x.size(1)

                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

                print("prepare data is finished!")
    
                patience = 20
                best = 1e9
                cnt_wait = 0
                
                if self.prompt_type == 'GPPT':
                    # initialize the GPPT hyperparametes via graph data
                    if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa','IMDB-MULTI','REDDIT-MULTI-5K']:

                        total_num_nodes = sum([data.num_nodes for data in train_dataset])
                        train_node_ids = torch.arange(0,total_num_nodes).squeeze().to(self.device)
                        self.gppt_loader = DataLoader(processed_dataset.to_data_list(), batch_size=1, shuffle=False)
                        for i, batch in enumerate(self.gppt_loader):
                            if(i==0):
                                node_for_graph_labels = torch.full((1,batch.x.shape[0]), batch.y.item())
                                
                                batch.x = self.embedding(batch.x.to(self.device))
                                
                                node_embedding = self.gnn(batch.x, batch.edge_index.to(self.device))
                            else:                   
                                node_for_graph_labels = torch.concat([node_for_graph_labels,torch.full((1,batch.x.shape[0]), batch.y.item())],dim=1)
                                
                                batch.x = self.embedding(batch.x.to(self.device))
                                
                                node_embedding = torch.concat([node_embedding,self.gnn(batch.x, batch.edge_index.to(self.device))],dim=0)
                        
                        node_for_graph_labels=node_for_graph_labels.reshape((-1)).to(self.device)
                        self.prompt.weigth_init(node_embedding,processed_dataset.edge_index.to(self.device), node_for_graph_labels, train_node_ids)

                        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)    
                        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)                
                    else:
                        train_node_ids = torch.arange(0,train_dataset.x.shape[0]).squeeze().to(self.device)
                        iterate_id_num = 0
                        for index, g in enumerate(train_dataset):
                            current_node_ids = iterate_id_num+torch.arange(0,g.x.shape[0]).squeeze().to(self.device)
                            iterate_id_num += g.x.shape[0]
                            previous_node_num = sum([self.dataset[i].x.shape[0] for i in range(idx_train[index]-1)])
                            train_node_ids[current_node_ids] += previous_node_num

                        self.gppt_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
                        for i, batch in enumerate(self.gppt_loader):
                            if(i==0):
                                node_for_graph_labels = torch.full((1,batch.x.shape[0]), batch.y.item())
                            else:                   
                                node_for_graph_labels = torch.concat([node_for_graph_labels,torch.full((1,batch.x.shape[0]), batch.y.item())],dim=1)
                        
                        embedding_x = self.embedding(self.dataset.x.to(self.device))

                        node_embedding = self.gnn(embedding_x, self.dataset.edge_index.to(self.device))
                        node_for_graph_labels=node_for_graph_labels.reshape((-1)).to(self.device)
                        self.prompt.weigth_init(node_embedding,self.dataset.edge_index.to(self.device), node_for_graph_labels, train_node_ids)

                        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)



                for epoch in range(1, self.epochs + 1):
                    t0 = time.time()

                    if self.prompt_type == 'None':
                        loss = self.Train(train_loader)
                    elif self.prompt_type == 'All-in-one':
                        loss = self.AllInOneTrain(train_loader,self.answer_epoch,self.prompt_epoch)
                    elif self.prompt_type in ['GSFP', 'GSmFP']:
                        loss = self.GSFPTrain(train_loader)
                    elif self.prompt_type =='Gprompt':
                        loss, center = self.GpromptTrain(train_loader)
                    elif self.prompt_type =='GPPT':
                        loss = self.GPPTtrain(train_loader)
                            
                    if loss < best:
                        best = loss  
                        cnt_wait = 0
                 
                    else:
                        cnt_wait += 1
                        if cnt_wait == patience:
                                print('-' * 100)
                                print('Early stopping at '+str(epoch) +' eopch!')
                                break
                    print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f}  ".format(epoch, time.time() - t0, loss))
                import math
                if not math.isnan(loss):
                    batch_best_loss.append(loss)
                print('Bengin to evaluate')
                
                if self.prompt_type == 'None':
                    test_acc, f1, roc, prc = GNNGraphEva(test_loader, self.gnn, self.answering,self.embedding, self.output_dim, self.device)
                    val_acc, val_f1, val_roc, val_prc = GNNGraphEva(val_loader, self.gnn, self.answering,self.embedding, self.output_dim, self.device)
                elif self.prompt_type =='GPPT':
                    test_acc, f1, roc, prc = GPPTGraphEva(test_loader,self.embedding, self.gnn, self.prompt, self.output_dim, self.device)
                    val_acc, val_f1, val_roc, val_prc = GPPTGraphEva(val_loader, self.embedding,self.gnn, self.prompt, self.output_dim, self.device)
                elif self.prompt_type == 'All-in-one':
                    test_acc, f1, roc, prc = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering,self.embedding, self.output_dim, self.device)
                    val_acc, val_f1, val_roc, val_prc = AllInOneEva(val_loader, self.prompt, self.gnn, self.answering,self.embedding, self.output_dim, self.device)
                elif self.prompt_type in ['GSFP', 'GSmFP']:
                    test_acc, f1, roc, prc = GSFPEva(test_loader, self.gnn, self.prompt, self.answering,self.embedding, self.output_dim, self.device)
                    val_acc, val_f1, val_roc, val_prc = GSFPEva(val_loader, self.gnn, self.prompt, self.answering,self.embedding, self.output_dim, self.device)
                elif self.prompt_type =='Gprompt':
                    test_acc, f1, roc, prc = GpromptEva(test_loader, self.embedding,self.gnn, self.prompt, center, self.output_dim, self.device)
                    val_acc, val_f1, val_roc, val_prc = GpromptEva(val_loader,self.embedding, self.gnn, self.prompt, center, self.output_dim, self.device)

 

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



            print(self.pre_train_type, self.gnn_type, self.prompt_type, " Graph Task completed")
            mean_best = np.mean(batch_best_loss)

            return  mean_best, mean_test_acc, std_test_acc, mean_f1, std_f1, mean_roc, std_roc, mean_prc, std_prc,mean_val_acc,std_val_acc

        

