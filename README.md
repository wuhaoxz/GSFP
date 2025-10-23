

# Reliable and Compact Graph Fine-tuning via Graph Sparse Prompting



This is the **code implementation** of the paper [Reliable and Compact Graph Fine-tuning via Graph Sparse Prompting](https://arxiv.org/abs/2410.21749). Our code is developed based on [ProG](https://github.com/sheldonresearch/ProG).



## Environment

```bash
python   3.10.14
torch   2.1.2+cu118
torch-cluster   1.6.3+pt21cu118
torch-geometric   2.3.0
torch-scatter   2.1.2+pt21cu118
torch-sparse   0.6.18+pt21cu118
torch-spline-conv   1.2.2+pt21cu118
torchaudio   2.1.2+cu118
torchmetrics   1.4.0.post0
torchsummary   1.5.1
torchvision   0.16.2+cu118
numpy   1.26.4
pandas   2.2.2
scikit-learn   1.5.1
scipy   1.14.0
```

 

## Pre-training

Run the following command:

```bash
python pre_train.py  --pretrain_lr 0.001 --pretrain_decay 0.00001 --pretrain_patience 100 --task GraphCL --dataset_name 'Flickr' --gnn_type 'GCN' --hid_dim 256 --num_layer 3 --epochs 100 --seed 42 --device 0
```

Use `--task` to switch the pre-training strategy, `--dataset_name` to change the dataset, and `--gnn_type` to change the GNN backbone. Besides, this [link](https://drive.google.com/drive/folders/1nGSlkUuYp3zvNlVJ-zfmo9Hi8c5SoFm0?usp=drive_link) provides pre-trained models.



## Node Classification

Example on the **Cora** dataset:

```bash
#GSFP
python downstream_task.py --lambda_ 0.000002 --pre_train_model_path './Experiment/pre_trained_model/Flickr/GraphCL.GCN.256hidden_dim.pth' --task NodeTask --dataset_name 'Cora' --gnn_type 'GCN' --prompt_type 'GSFP' --shot_num 1 --hid_dim 256 --num_layer 3  --lr 0.001 --decay 5e-4 --seed 42 --device 0 --batch_size 128

#GSmFP
python downstream_task.py --lambda_ 0.07 --pnum 10 --pre_train_model_path './Experiment/pre_trained_model/Flickr/GraphCL.GCN.256hidden_dim.pth' --task NodeTask --dataset_name 'Cora' --gnn_type 'GCN' --prompt_type 'GSmFP' --shot_num 1 --hid_dim 256 --num_layer 3  --lr 0.001 --decay 5e-4 --seed 42 --device 0 --batch_size 128
```

Use `--pre_train_model_path` to change the pre-trained model, `--dataset_name` to change the dataset, and `--shot_num` to change the number of shots.
 The parameter `--lambda_` controls the sparsity. Note that when `lambda` is 0, GSFP and GSmFP degrade to [GPF](https://arxiv.org/abs/2209.15240) and [GPF-plus](https://arxiv.org/abs/2209.15240), respectively.



## Graph Classification

Example on the **COX2** dataset:

```bash
#GSFP
python downstream_task.py --lambda_ 0.00044 --pre_train_model_path './Experiment/pre_trained_model/DD/GraphCL.GCN.256hidden_dim.pth' --task GraphTask --dataset_name 'COX2' --gnn_type 'GCN' --prompt_type 'GSFP' --shot_num 1 --hid_dim 256 --num_layer 3  --lr 0.001 --decay 5e-4 --seed 42 --device 0 --batch_size 128

#GSmFP
python downstream_task.py --lambda_ 0.0022 --pnum 20 --pre_train_model_path './Experiment/pre_trained_model/DD/GraphCL.GCN.256hidden_dim.pth' --task GraphTask --dataset_name 'COX2' --gnn_type 'GCN' --prompt_type 'GSmFP' --shot_num 1 --hid_dim 256 --num_layer 3  --lr 0.001 --decay 5e-4 --seed 42 --device 0 --batch_size 128
```





