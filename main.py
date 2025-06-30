from train import training
from model import MultiMixedInformationNetwork
from tool import set_log
import torch
from data import MoleDataSet
from graph import GraphOne,GraphBatch
num_folds=10
log_path="save_model/log"
log=set_log("Train",save_path=log_path)
dataset_path="Process_Dataset\\Process_BBBP.csv"
dataset_type="classification"#classification or regression
seed=[0,1,2,3,4,5,6,7,8,9]
val_path=None
test_path=None
split=[0.8,0.1,0.1]
split_type="scaffold"#random or scaffold
metric="auc"#rmse or auc
save_path="model_for_atom_similarity\\BBBP_model"
batch_size=64
init_lr=1e-4
warmup_epochs=5
total_epochs=50
task_name=["1"]
task_num=1
max_lr=1e-4
final_lr=1e-4
fp_Linear_dim=512
dropout_FPN=0.05
cuda="True"
hidden_size=300
fp_type="mixed"
hidden_dim=1024
GAT_ratio=0.5
in_features=57
bond_features_dim=13
out_features=110#
dropout_gnn_ratio=0.5#
leaky_alpha=0.01
elu_alpha=1.0
N=3#
h=19#
d_k=96#
d_model=d_k*h
dropout_attn=0.5#
lambda_attention=0.5
trainable_lambda="True"
N_dense=2
scale_norm="DyT"#LN or DyT
number_layer=2#

score_average,score_standard=training(num_folds,log,dataset_path,dataset_type,seed,val_path,test_path,split,split_type,metric,save_path,batch_size,
                                      init_lr,warmup_epochs,total_epochs,max_lr,final_lr,task_name,task_num,fp_Linear_dim,dropout_FPN,cuda,hidden_size,fp_type,hidden_dim,GAT_ratio,
                                      in_features,bond_features_dim,out_features,dropout_gnn_ratio,leaky_alpha,elu_alpha,N,h,dropout_attn,d_model,lambda_attention,trainable_lambda,N_dense,scale_norm,number_layer)
