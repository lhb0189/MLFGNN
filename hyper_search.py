import csv
from hyperopt import fmin,hp,tpe
import os
from copy import deepcopy
from train import training
from model import MultiMixedInformationNetwork
from tool import set_log
space={
    "dropout_FPN":hp.quniform('dropout_FPN',low=0.0,high=0.6,q=0.05),
    "N":hp.quniform('N',low=2,high=6,q=1),
    "dropout_gnn_ratio":hp.quniform('dropout_gnn_ratio',low=0.0,high=0.6,q=0.05),
    "out_features":hp.quniform('out_features',low=80,high=160,q=10),
    "dropout_attn":hp.quniform('dropout_attn',low=0.0,high=0.6,q=0.05),
    "h":hp.quniform('h',low=12,high=20,q=1),
    "d_k":hp.quniform('d_k',low=64,high=128,q=8),
}
def fn(space):
    dropout_FPN= space['dropout_FPN']
    N=space['N']
    number_layer = space['number_layer']
    dropout_gnn_ratio = space['dropout_gnn_ratio']
    out_features = space['out_features']
    dropout_attn = space['dropout_attn']
    h=space['h']
    d_k=space['d_k']
    N=int(N)
    number_layer=int(number_layer)
    out_features=int(out_features)
    h=int(h)
    d_k=int(d_k)
    num_folds=5
    log_path="hyperoptimize/log"
    log=set_log("Train",save_path=log_path)
    result_path=os.path.join(log_path,"hyper_para_result.txt")
    dataset_path="Process_Lipophilicity.csv"
    dataset_type="regression"
    seed=42
    val_path = None
    test_path = None
    split = [0.8, 0.1, 0.1]
    split_type = "random"
    metric = "rmse"
    save_path="hyperoptimize/model"
    batch_size=64
    init_lr=1e-4
    warmup_epochs=5
    total_epochs=50
    max_lr=1e-3
    final_lr=1e-4
    task_name="Lipophilicity_model"
    task_num=1
    fp_Linear_dim=512
    cuda="True"
    hidden_size=300
    fp_type="mixed"
    hidden_dim = 1024
    GAT_ratio = 0.5
    in_features = 57
    bond_features_dim = 13
    leaky_alpha=0.1
    elu_alpha = 1.0
    d_model = d_k * h
    lambda_attention = 0.5
    trainable_lambda = "True"
    N_dense = 2
    scale_norm = "DyT"
    ave,std=training(num_folds,log,dataset_path,dataset_type,seed,val_path,test_path,split,split_type,metric,save_path,batch_size,
                                      init_lr,warmup_epochs,total_epochs,max_lr,final_lr,task_name,task_num,fp_Linear_dim,dropout_FPN,cuda,hidden_size,fp_type,hidden_dim,GAT_ratio,in_features,bond_features_dim,out_features,dropout_gnn_ratio,leaky_alpha,elu_alpha,N,h,dropout_attn,d_model,lambda_attention,trainable_lambda,N_dense,scale_norm,number_layer)
    with open(result_path,'a') as file:
        file.write(str(space)+'\n')
        file.write("dropout_FPN:"+str(dropout_FPN)+","+"N:"+str(N)+","+"number_layer:"+str(number_layer)+","+"dropout_gnn_ratio:"+str(dropout_gnn_ratio)+","+"out_features:"+str(out_features)+","+"dropout_attn:"+str(dropout_attn)+","+"h:"+str(h)+","+"d_K:"+str(d_k)+"."+'Result '+str(metric)+' : '+str(ave)+' +/- '+str(std)+'\n')#将本轮超参数与模型评估结果写入文件
    if ave is None:
        if dataset_type == 'classification':
            ave = 0
        else:
            raise ValueError('Result of BACE_model is error.')
    if dataset_type == 'classification':
        return -ave
    else:
        return ave
def hyper_searching(search_num):
    log_path="hyperoptimize/log"
    result_path=os.path.join(log_path,"hyper_para_result.txt")
    result=fmin(fn,space,tpe.suggest,search_num)
    with open(result_path,'a') as file:
        file.write("Best hyper-parameters : \n ")
        file.write(str(result)+"\n")
hyper_searching(20)
