import os
import csv
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tool import mkdir, get_task_name, load_data, split_data, get_label_scaler, get_loss, get_metric, save_model, load_model
from model import MultiMixedInformationNetwork,get_mask
from graph import create_graph,GraphOne,GraphBatch
from data import MoleDataSet
import copy
import random
def epoch_train(model,dataset,loss_f,optimizer,seed,batch_size):
    model.train()
    dataset.random_data(seed)
    loss_sum=0#记录训练的时候损失值
    data_used=0
    Batch_size=batch_size
    for i in range(0,len(dataset),Batch_size):
        if data_used + Batch_size > len(dataset):
            data_now=MoleDataSet(dataset[i:])
        else:
            data_now=MoleDataSet(dataset[i:i+Batch_size])
        smiles=data_now.smile()
        label=data_now.label()
        Graph_data=create_graph(smiles)
        atom_features, atom_index, bond_features = Graph_data.get_feature()
        bond_features = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in bond_features.items()}
        atom_features = atom_features.to("cuda")
        adjacency_matrix = Graph_data.get_adjacency_matrix()
        adjacency_matrix = adjacency_matrix.to("cuda")
        mask_matrix = get_mask(Graph_data)
        mask_matrix = mask_matrix.to("cuda")
        target = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])
        target = target.to("cuda")
        model.zero_grad()
        pred,FPN_output,Graph_output,averaged_tensor,Transformer_features,new_embeddings=model(smiles,atom_features,mask_matrix,adjacency_matrix,bond_features,atom_index)#CrossAttention的forward参数
        loss=loss_f(pred,target)
        loss=loss.sum()/pred.size(0)
        loss_sum+=loss.item()
        data_used+=len(smiles)
        loss.backward()
        optimizer.step()
def predict(model,dataset,batch_size,scaler):
    model.eval()
    pred=[]
    data_total=len(dataset)
    for i in range(0,data_total,batch_size):
        data_now=MoleDataSet(dataset[i:i+batch_size])
        smiles=data_now.smile()
        Graph_data=create_graph(smiles)
        atom_features, atom_index, bond_features = Graph_data.get_feature()
        bond_features = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in bond_features.items()}
        atom_features = atom_features.to("cuda")
        adjacency_matrix = Graph_data.get_adjacency_matrix()
        adjacency_matrix = adjacency_matrix.to("cuda")
        mask_matrix = get_mask(Graph_data)
        mask_matrix = mask_matrix.to("cuda")
        with torch.no_grad():
            pred_now,FPN_output,Graph_output,averaged_tensor,Transformer_features,new_embeddings=model(smiles,atom_features,mask_matrix,adjacency_matrix,bond_features,atom_index)
        pred_now=pred_now.data.cpu().numpy()
        if scaler is not None:
            ave=scaler[0]
            std=scaler[1]
            pred_now=np.array(pred_now).astype(float)
            change_1=pred_now*std+ave
            pred_now=np.where(np.isnan(change_1),None,change_1)
        pred_now=pred_now.tolist()
        pred.extend(pred_now)
    return pred
def compute_score(pred,label,metric_f,log,task_num,dataset_type):
    info=log.info
    if len(pred)==0:
        return [float('nan')] * task_num
    pred_val=[]
    label_val=[]
    for i in range(task_num):
        pred_val_i=[]
        label_val_i=[]
        for j in range(len(pred)):
            if label[j][i] is not None:
                pred_val_i.append(pred[j][i])
                label_val_i.append(label[j][i])
        pred_val.append(pred_val_i)
        label_val.append(label_val_i)
    result=[]
    for i in range(task_num):
        if dataset_type == 'classification':
            if all(one == 0 for one in label_val[i]) or all(one == 1 for one in label_val[i]):
                info('Warning: All labels are 1 or 0.')
                result.append(float('nan'))
                continue
            if all(one == 0 for one in pred_val[i]) or all(one == 1 for one in pred_val[i]):
                info('Warning: All predictions are 1 or 0.')
                result.append(float('nan'))
                continue
        re=metric_f(label_val[i],pred_val[i])
        result.append(re)
    return result
def fold_train(log,dataset_path,dataset_type,seed,val_path,test_path,split,split_type,metric,model,save_path,batch_size,
               init_lr,warmup_epochs,total_epochs,max_lr,final_lr,task_name):
    info=log.info
    debug = log.debug
    debug("Start loading data")
    dataset=load_data(dataset_path)
    task_num=dataset.task_num()
    debug(f'Splitting dataset with Seed = {seed}.')
    if val_path !=None:
        val_data=load_data(val_path)
    if test_path!=None:
        test_data=load_data(test_path)
    if val_path!=None and test_path!=None:
        train_data=dataset
    elif val_path!=None:
        split_ratio=(split[0],0,split[2])
        train_data,_,test_data=split_data(dataset,split_type,split_ratio,seed,log)
    elif test_path!=None:
        split_ratio=(split[0],split[1],0)
        train_data,val_data,_=split_data(dataset,split_type,split_ratio,seed,log)
    else:
        train_data,val_data,test_data=split_data(dataset,split_type,split,seed,log)
    debug(
        f'Dataset size: {len(dataset)}    Train size: {len(train_data)}    Val size: {len(val_data)}    Test size: {len(test_data)}')  # 输出三种数据集的数量
    if dataset_type=="regression":
        label_scaler=get_label_scaler(train_data)
    else:
        label_scaler=None
    train_data_size=len(train_data)
    loss_f=get_loss(dataset_type)
    metric_f=get_metric(metric)
    debug("Training Model")
    debug(model)
    new_model=copy.deepcopy(model).to(torch.device("cuda"))
    model=model.to(torch.device("cuda"))
    save_model(os.path.join(save_path,"BACE_model.pt"),model,label_scaler)
    optimizer=Adam(params=model.parameters(), lr=init_lr,weight_decay=1e-3)
    if dataset_type=="classification":
        best_score=-float("inf")
    else:
        best_score=float("inf")
    best_epoch=0
    for epoch in range(total_epochs):
        info(f'Epoch {epoch}')
        epoch_train(model,train_data,loss_f,optimizer,seed,batch_size)
        train_pred=predict(model,train_data,batch_size,label_scaler)
        train_label=train_data.label()
        train_score=compute_score(train_pred,train_label,metric_f,log,task_num,dataset_type)
        val_pred=predict(model,val_data,batch_size,label_scaler)
        val_label=val_data.label()
        val_score=compute_score(val_pred,val_label,metric_f,log,task_num,dataset_type)
        average_train_score=np.nanmean(train_score)
        info(f'Train{metric}={average_train_score:.6f}')
        average_val_score=np.nanmean(val_score)
        info(f'Validation{metric}={average_val_score:.6f}')
        test_label = test_data.label()
        test_pred = predict(model, test_data, batch_size, label_scaler)
        test_score = compute_score(test_pred, test_label, metric_f, log, task_num, dataset_type)
        average_test_score = np.nanmean(test_score)
        info(f"Seed {seed} :test{metric}={average_test_score:.6f}")
        if task_num>1:
            for one_name,one_score in zip(task_name,test_score):
                info(f'test {one_name} {metric}= {one_score:.6f}')
        if dataset_type=="classification" and average_val_score>best_score:
            best_score=average_val_score
            best_epoch=epoch
            save_model(os.path.join(save_path,'BACE_model.pt'),model,label_scaler)
        elif dataset_type=="regression" and average_val_score<best_score:
            best_score=average_val_score
            best_epoch=epoch
            save_model(os.path.join(save_path,'BACE_model.pt'),model,label_scaler)
    info(f"Best validation {metric}={best_score:.6f} on epoch {best_epoch}")
    model=load_model(new_model,os.path.join(save_path,"BACE_model.pt"),cuda="True",log=log)
    test_label=test_data.label()
    test_pred=predict(model,test_data,batch_size,label_scaler)
    test_score=compute_score(test_pred,test_label,metric_f,log,task_num,dataset_type)
    average_test_score=np.nanmean(test_score)
    info(f"Seed {seed} :test{metric}={average_test_score:.6f}")
    if task_num>1:
        for one_name,one_score in zip(task_name,test_score):
            info(f'task {one_name}{metric}={one_score:.6f}')
    return test_score
def training(num_folds,log,dataset_path,dataset_type,seed,val_path,test_path,split,split_type,metric,save_path,batch_size,
             init_lr,warmup_epochs,total_epochs,max_lr,final_lr,task_names,task_num,
             fp_Linear_dim, dropout_FPN, cuda, hidden_size, fp_type,
             hidden_dim, GAT_ratio, in_features, bond_features_dim, out_features,
             dropout_gnn_ratio, leaky_alpha, elu_alpha, N, h, dropout_attn,
             d_model, lambda_attention, trainable_lambda, N_dense, scale_norm,
             number_layer):
    seed_list=seed
    info=log.info
    score=[]
    for num_fold in range(num_folds):
        new_seed=seed_list[num_fold]
        info(f"Seed{new_seed}")
        torch.manual_seed(new_seed)
        np.random.seed(new_seed)
        random.seed(new_seed)
        model = MultiMixedInformationNetwork(dataset_type, task_num, fp_Linear_dim, dropout_FPN, cuda, hidden_size,
                                             fp_type,
                                             hidden_dim, GAT_ratio, in_features, bond_features_dim, out_features,
                                             dropout_gnn_ratio, leaky_alpha, elu_alpha, N, h, dropout_attn, dropout_FPN,
                                             d_model, lambda_attention, trainable_lambda, N_dense, scale_norm,
                                             number_layer,
                                             number_layer)
        save_path_seed=os.path.join(save_path,f'Seed_{new_seed}')
        mkdir(save_path_seed)
        fold_score=fold_train(log,dataset_path,dataset_type,new_seed,val_path,test_path,split,split_type,metric,model,save_path_seed,batch_size,
                              init_lr,warmup_epochs,total_epochs,max_lr,final_lr,task_names)
        score.append(fold_score)
    score=np.array(score)
    info(f"Running {num_folds} fold in total.")
    if num_folds>1:
        for num_fold , fold_score in enumerate(score):
            seed_1 =seed_list[num_fold]
            info(f'Seed {seed_1}:test{metric}={np.nanmean(fold_score):.6f}')
            if task_num>1:
                for one_name,one_score in zip(task_names,fold_score):
                    info(f"Task{one_name}{metric}={one_score:.6f}")
    average_task_score=np.nanmean(score,axis=1)
    score_average=np.nanmean(average_task_score)
    score_std=np.nanstd(average_task_score)
    info(f"Average test{metric} = {score_average:.6f} +/- {score_std:.6f}")
    if task_num>1:
        for i,one_name in enumerate(task_names):
            info(f'average all fold {one_name}{metric}={np.nanmean(score[:,i]):.6f}+/-{np.nanstd(score[:,i]):.6f}')
    return score_average,score_std