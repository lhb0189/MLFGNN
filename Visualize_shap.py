import torch
from model import MultiMixedInformationNetwork,get_mask
from tool import load_model,split_data,load_data
import numpy as np
import random
from data import MoleDataSet,MoleData
from graph import GraphOne,GraphBatch,create_graph
from sklearn.linear_model import LinearRegression
import shap
import matplotlib.pyplot as plt
from tool import set_log
log_path="model_for_Visualize/log"
log=set_log("Train",save_path=log_path)
seed=11
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
dataset_type="classification"
task_num=27
fp_Linear_dim=512
dropout_FPN=0.05
cuda="True"
hidden_size=300
fp_type="mixed"
hidden_dim=1024
GAT_ratio=0.5
in_features=57
bond_features_dim=13
out_features=110
dropout_gnn_ratio=0.5
leaky_alpha=0.01
elu_alpha=1.0
N=3
h=19
d_k=96
d_model=d_k*h
dropout_attn=0.5
lambda_attention=0.5
trainable_lambda="True"
N_dense=2
scale_norm="DyT"
number_layer=2
split_type="random"
split=[0.8,0.1,0.1]
path="model_for_Visualize\\SIDER_model\\Seed_29\\BACE_model.pt"
new_model = MultiMixedInformationNetwork(dataset_type, task_num, fp_Linear_dim, dropout_FPN, cuda, hidden_size,
                                         fp_type,
                                         hidden_dim, GAT_ratio, in_features, bond_features_dim, out_features,
                                         dropout_gnn_ratio, leaky_alpha, elu_alpha, N, h, dropout_attn, dropout_FPN,
                                         d_model, lambda_attention, trainable_lambda, N_dense, scale_norm,
                                         number_layer,
                                         number_layer)
model=load_model(new_model,path,"True",log)
model.eval()
FP_features=[]
Graph_features=[]
predictions=[]
dataset_path="Process_Dataset\\Process_SIDER.csv"
dataset=load_data(dataset_path)
batch_size=32
train_data,val_data,test_data=split_data(dataset,split_type,split,seed,log)
train_data_total=len(train_data)
val_data_total=len(val_data)
test_data_total=len(test_data)
total_data=train_data_total+val_data_total+test_data_total
for i in range(0,total_data,batch_size):
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
        output,FPN_output,Graph_output=model(smiles,atom_features,mask_matrix,adjacency_matrix,bond_features,atom_index)
    FP_features.append(FPN_output.detach().cpu().numpy())
    Graph_features.append(Graph_output.detach().cpu().numpy())
    predictions.append(output.detach().cpu().numpy())
X_fp = np.concatenate(FP_features, axis=0)
X_graph = np.concatenate(Graph_features, axis=0)
X=np.concatenate([X_fp,X_graph],axis=1)
y = np.concatenate(predictions, axis=0).reshape(-1)
reg = LinearRegression()
reg.fit(X, y)
explainer = shap.Explainer(reg.predict, X)
shap_values = explainer(X)
shap_values_fp = shap_values.values[:, :X_fp.shape[1]]
shap_values_graph = shap_values.values[:, X_fp.shape[1]:]
fp_contrib = np.mean(np.abs(shap_values_fp))
graph_contrib = np.mean(np.abs(shap_values_graph))
labels = ['Fingerprint', 'Molecular Graph']
values = [fp_contrib, graph_contrib]
plt.figure(figsize=(6, 4))
plt.bar(labels, values)
plt.ylabel("Average absolute SHAP value")
plt.title("Contribution to Final Prediction")
plt.tight_layout()
plt.show()