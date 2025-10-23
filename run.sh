#pretrain
python pre_train.py  --pretrain_lr 0.001 --pretrain_decay 0.00001 --pretrain_patience 100 --task GraphCL --dataset_name 'Flickr' --gnn_type 'GCN' --hid_dim 256 --num_layer 3 --epochs 100 --seed 42 --device 0

##Cora
#GCN
python downstream_task.py --pre_train_model_path 'None' --task NodeTask --dataset_name 'Cora' --gnn_type 'GCN' --prompt_type 'None' --shot_num 1 --hid_dim 256 --num_layer 3  --lr 0.001 --decay 5e-4 --seed 42 --device 0 --batch_size 128
#FT
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Flickr/GraphCL.GCN.256hidden_dim.pth' --task NodeTask --dataset_name 'Cora' --gnn_type 'GCN' --prompt_type 'None' --shot_num 1 --hid_dim 256 --num_layer 3  --lr 0.001 --decay 5e-4 --seed 42 --device 0 --batch_size 128
#GSFP
python downstream_task.py --lambda_ 0.000002 --pre_train_model_path './Experiment/pre_trained_model/Flickr/GraphCL.GCN.256hidden_dim.pth' --task NodeTask --dataset_name 'Cora' --gnn_type 'GCN' --prompt_type 'GSFP' --shot_num 1 --hid_dim 256 --num_layer 3  --lr 0.001 --decay 5e-4 --seed 42 --device 0 --batch_size 128
#GSmFP
python downstream_task.py --lambda_ 0.07 --pnum 10 --pre_train_model_path './Experiment/pre_trained_model/Flickr/GraphCL.GCN.256hidden_dim.pth' --task NodeTask --dataset_name 'Cora' --gnn_type 'GCN' --prompt_type 'GSmFP' --shot_num 1 --hid_dim 256 --num_layer 3  --lr 0.001 --decay 5e-4 --seed 42 --device 0 --batch_size 128

##COX2
#GCN
python downstream_task.py --pre_train_model_path 'None' --task GraphTask --dataset_name 'COX2' --gnn_type 'GCN' --prompt_type 'None' --shot_num 1 --hid_dim 256 --num_layer 3  --lr 0.001 --decay 5e-4 --seed 42 --device 0 --batch_size 128
#FT
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/DD/GraphCL.GCN.256hidden_dim.pth' --task GraphTask --dataset_name 'COX2' --gnn_type 'GCN' --prompt_type 'None' --shot_num 1 --hid_dim 256 --num_layer 3  --lr 0.001 --decay 5e-4 --seed 42 --device 0 --batch_size 128
#GSFP
python downstream_task.py --lambda_ 0.00044 --pre_train_model_path './Experiment/pre_trained_model/DD/GraphCL.GCN.256hidden_dim.pth' --task GraphTask --dataset_name 'COX2' --gnn_type 'GCN' --prompt_type 'GSFP' --shot_num 1 --hid_dim 256 --num_layer 3  --lr 0.001 --decay 5e-4 --seed 42 --device 0 --batch_size 128
#GSmFP
python downstream_task.py --lambda_ 0.0022 --pnum 20 --pre_train_model_path './Experiment/pre_trained_model/DD/GraphCL.GCN.256hidden_dim.pth' --task GraphTask --dataset_name 'COX2' --gnn_type 'GCN' --prompt_type 'GSmFP' --shot_num 1 --hid_dim 256 --num_layer 3  --lr 0.001 --decay 5e-4 --seed 42 --device 0 --batch_size 128

