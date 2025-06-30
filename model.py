import math,copy
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from graph import create_graph,GraphOne,GraphBatch
from PubChemFP import GetPubChemFPs
from rdkit.Chem import AllChem,MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
import torch.nn.functional as F
from dynamic_tanh import DynamicTanh
def smiles_to_ecfps(smiles,radius=2,nBits=1024):#ECFP
    mol=Chem.MolFromSmiles(smiles)
    ecfps_gen=rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=nBits)
    ecfps_fp=ecfps_gen.GetFingerprintAsNumPy(mol)
    return torch.tensor(ecfps_fp).view(1,-1)#1*1024
def smiles_to_MACCS(smiles):
    mol=Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return torch.tensor(fp).view(1,-1)#1*167
def smiles_to_PubChem(smiles):
    mol=Chem.MolFromSmiles(smiles)
    mol2 = Chem.AddHs(mol)
    return torch.tensor(GetPubChemFPs(mol2)).view(1,-1)#1*881
def smiles_to_Pharmacophere(smiles):
    mol=Chem.MolFromSmiles(smiles)
    fp_phaErGfp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
    return torch.tensor(fp_phaErGfp).view(1,-1)#1*441
class FPN(nn.Module):#Fingerprints Network
    def __init__(self,fp_Linear_dim,dropout_FPN,cuda,hidden_size,fp_type):
        super(FPN,self).__init__()
        self.fp_Linear_dim=fp_Linear_dim
        self.dropout_fpn =dropout_FPN
        self.cuda=cuda
        self.hidden_dim=hidden_size
        self.fp_type=fp_type
        if self.fp_type=='mixed':
            self.fp_initial_dim=2346
        else:
            self.fp_initial_dim=1024
        self.fc1=nn.Linear(self.fp_initial_dim,self.fp_Linear_dim).to("cuda")
        self.batchNorm1=nn.BatchNorm1d(self.fp_Linear_dim).to("cuda")
        self.act_func=nn.ReLU().to("cuda")
        self.Dropout=nn.Dropout(p=self.dropout_fpn).to("cuda")
        self.fc2=nn.Linear(self.fp_Linear_dim,self.hidden_dim).to("cuda")
        self.batchNorm2=nn.BatchNorm1d(self.hidden_dim).to("cuda")
        self.dropout=nn.Dropout(p=self.dropout_fpn).to("cuda")
    def forward(self,smiles):#[batch_size,smiles]
        smiles_to_fp=[]
        for smile in smiles:
            if self.fp_type=="mixed":
                fp_morgan=smiles_to_ecfps(smile)
                fp_pubChem=smiles_to_PubChem(smile)
                fp_pharmacophere=smiles_to_Pharmacophere(smile)
                fp_list = torch.cat([fp_morgan, fp_pubChem, fp_pharmacophere], dim=1)
                fp_list =fp_list.squeeze(0)
            else:
                fp_morgan=smiles_to_ecfps(smile,radius=2,nBits=1024)
                fp_list=fp_morgan
            smiles_to_fp.append(fp_list)
        fp_list=torch.stack(smiles_to_fp,dim=0)
        fp_list=fp_list.to(torch.float32)
        if self.cuda:
            fp_list=fp_list.cuda()
        fpn_output=self.fc1(fp_list)#[batch_size,2346]->[batch_size,fp_Linear_dim]
        fpn_output=self.dropout(fpn_output)
        fpn_output=self.act_func(fpn_output)
        fpn_output=self.fc2(fpn_output)#[batch_size,fp_Linear_dim]->[batch_size,hidden_size]
        return fpn_output
class GATlayerV1(nn.Module):
    def __init__(self,in_features,bond_features_dim,output_features,dropout_gnn_ratio,leaky_alpha,elu_alpha):
        super(GATlayerV1,self).__init__()
        self.bond_features_dim=bond_features_dim
        self.dropout_gnn=nn.Dropout(p=dropout_gnn_ratio)
        self.in_features=in_features
        self.out_features=output_features
        self.atom_fc1=nn.Linear(self.in_features,self.out_features).to("cuda")
        self.atom_fc2=nn.Linear(self.in_features+self.bond_features_dim,self.out_features).to("cuda")
        self.W=nn.Linear(2*self.out_features,1).to("cuda")
        self.leaky_alpha=leaky_alpha
        self.elu_alpha=elu_alpha
        self.attend=nn.Linear(output_features,output_features).to("cuda")
        self.GRUcell=nn.GRUCell(self.out_features,self.out_features).to("cuda")
    def forward(self,atom_features,bond_features):
        N=atom_features.shape[0]
        atom_features=atom_features.to("cuda")
        new_atom_features=F.leaky_relu(self.atom_fc1(atom_features),negative_slope=self.leaky_alpha)
        neighbor_features_transform = self.attend(self.dropout_gnn(new_atom_features))
        new_list=[]
        for i in range(N):
            transform=torch.zeros(self.out_features)
            transform=transform.to("cuda")
            atom_neighbor_features = []
            index=[]
            for keys,values in bond_features.items():
                if keys[1]==i:
                    index.append(keys[0])
                    neighbor_atom_features=torch.cat([atom_features[keys[0]],values],dim=0)
                    atom_neighbor_features.append(neighbor_atom_features)
            l=[]
            if len(index)==0:
                new_list.append(atom_features[i])
            if len(atom_neighbor_features)==0:
                new_list.append(new_atom_features[i])
            else:
                stacked_tensor=torch.stack(atom_neighbor_features,dim=0)
                new_neighbor=F.leaky_relu(self.atom_fc2(stacked_tensor),negative_slope=self.leaky_alpha)
                for j in range(len(index)):
                    final_neighbor=torch.cat([new_atom_features[i],new_neighbor[j]],dim=0)
                    l.append(final_neighbor)
                final_embedding_1=torch.stack(l,dim=0)
                final_embedding_2=F.leaky_relu(self.W(self.dropout_gnn(final_embedding_1)),negative_slope=self.leaky_alpha)
                score=F.softmax(final_embedding_2,dim=0)
                score=score.to("cuda")
                neighbor_features_transform=neighbor_features_transform.to("cuda")
                for j in range(len(index)):
                    transform+=score[j]*neighbor_features_transform[index[j]]
                context=F.elu(transform,alpha=self.elu_alpha)
                new_list.append(context)
        output_embedding=torch.stack(new_list,dim=0)#[total_num_atoms,out_features]
        final_output_embedding=self.GRUcell(output_embedding,new_atom_features)
        return final_output_embedding,new_atom_features
class GATlayerV2(nn.Module):
    def __init__(self,output_features,dropout_gnn_ratio,leaky_alpha,elu_alpha):
        super(GATlayerV2,self).__init__()
        self.out_features=output_features
        self.dropout_gnn=nn.Dropout(p=dropout_gnn_ratio)
        self.leaky_alpha=leaky_alpha
        self.elu_alpha=elu_alpha
        self.atom_fc1=nn.Linear(output_features*2,1)
        self.atom_fc2=nn.Linear(output_features,output_features)
        self.GRUcell=nn.GRUCell(self.out_features,self.out_features)
    def forward(self,atom_features,bond_features):
        N=atom_features.shape[0]
        neighbor_features_transform =self.atom_fc2(self.dropout_gnn(atom_features))
        new_list = []
        for i in range(N):
            transform=torch.zeros(self.out_features)
            transform=transform.to("cuda")
            index=[]
            for keys,values in bond_features.items():
                if keys[1]==i:
                    index.append(keys[0])
            l=[]
            if len(index) == 0:
                new_list.append(atom_features[i])
            else:
                for j in range(len(index)):
                    final_neighbor=torch.cat([atom_features[i],atom_features[index[j]]],dim=0)
                    l.append(final_neighbor)
                final_embedding_1=torch.stack(l,dim=0)
                final_embedding_2=F.leaky_relu(self.atom_fc1(self.dropout_gnn(final_embedding_1)),negative_slope=self.leaky_alpha)
                score=F.softmax(final_embedding_2,dim=0)
                for j in range(len(index)):
                    transform+=score[j]*neighbor_features_transform[index[j]]
                context=F.elu(transform,alpha=self.elu_alpha)
                new_list.append(context)
        output_embedding=torch.stack(new_list,dim=0)
        final_output=self.GRUcell(output_embedding,atom_features)
        return final_output
def clones(module, N):
    #Produce N identical layers
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def get_mask(graph_batch_infor):
    #graph为graph_batch类
    length=graph_batch_infor.atom_no
    same_group=torch.full((length,length),-1e9)#[total_atom_number,total_atom_number]
    atom_index=graph_batch_infor.atom_index
    for index in atom_index:
        start,atom_number=index[0],index[1]
        same_group[start:start+atom_number,start:start+atom_number]=torch.zeros((atom_number,atom_number))
    return same_group
def graph_attention(query,key,value,mask,adjacency_matrix,lambdas,trainable_lambda,dropout=None):
    #q,k,v:[h,total_atom_number,d_k]
    d_k=query.size(-1)
    eps=1e-6
    scores=torch.matmul(query,key.transpose(-2,-1)) /math.sqrt(d_k)#[total_atom_number,total_atom_number]
    scores_shape=scores.shape
    mask_shape=mask.shape
    scores=scores+mask
    p_attn=F.softmax(scores,dim=-1).to("cuda")
    adj_matrix = adjacency_matrix / (adjacency_matrix.sum(dim=-1,keepdim=True)+ eps)#对邻接矩阵归一化(加上eps防止除以0)
    adj_matrix=adj_matrix.unsqueeze(0).repeat(query.shape[0],1,1)
    p_adj=adj_matrix.to("cuda")
    value=value.to("cuda")
    if trainable_lambda:
        softmax_attention, softmax_adjacency = lambdas.cuda()
        p_weighted = softmax_attention * p_attn + softmax_adjacency * p_adj
    else:
        lambda_attention, lambda_adjacency = lambdas
        p_weighted = lambda_attention * p_attn + lambda_adjacency * p_adj
    if dropout is not None:
        p_weighted = dropout(p_weighted)
    atom_features=torch.matmul(p_weighted,value).to("cuda")
    #atom_features:[total_num_atoms,d_k],p_weighted:[total_num_atoms,total_num_atoms]
    # p_attn:[total_num_atoms,total_num_atoms]
    return atom_features,p_weighted,p_attn
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout,lambda_attention,trainable_lambda):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h=h
        self.trainable_lambda=trainable_lambda
        if trainable_lambda:
            lambda_distance=1. -lambda_attention
            lambdas_tensor=torch.tensor([lambda_attention,lambda_distance],requires_grad=True)
            self.lambdas=torch.nn.Parameter(lambdas_tensor)
        else:
            lambdas_distance=1. - lambda_attention
            self.lambdas=(lambda_attention,lambdas_distance)
        self.linears=clones(nn.Linear(d_model, d_model).to("cuda"), 4)
        self.dropout=nn.Dropout(p=dropout)
    def forward(self,query,key,value,mask,adjacency_matrix):
        total_atom_number=query.size(0)
        query=query.to("cuda")
        key=key.to("cuda")
        value=value.to("cuda")
        q,k,v=[l(x).contiguous().view(total_atom_number,self.h,self.d_k).permute(1,0,2) for l,x in zip(self.linears,(query,key,value))]
        x,self.attn,self.self_attn=graph_attention(q,k,v,mask,adjacency_matrix,lambdas=self.lambdas,trainable_lambda=self.trainable_lambda,dropout=self.dropout)
        x=x.permute(1, 0, 2).contiguous().view(total_atom_number,self.h*self.d_k)
        return self.linears[-1](x)
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,N_dense,dropout=0.1,leaky_relu_slope=0.0,dense_output_nonlinearity="relu"):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = N_dense
        self.linears = clones(nn.Linear(d_model, d_model).to("cuda"), N_dense)
        self.dropout = clones(nn.Dropout(dropout).to("cuda"), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x
    def forward(self, x):
        if self.N_dense == 0:
            return x
        for i in range(len(self.linears)-1):
            x = self.dropout[i](F.leaky_relu(self.linears[i](x), negative_slope=self.leaky_relu_slope))
        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](x)))
class Embeddings(nn.Module):
    def __init__(self, d_model, d_atom, dropout):
        super(Embeddings, self).__init__()
        self.lin = nn.Linear(d_atom, d_model).to("cuda")
        self.dropout = nn.Dropout(dropout).to("cuda")
    def forward(self, x):
        return self.dropout(self.lin(x))
class SublayerConnection(nn.Module):#作残差连接
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, atom_dim, dropout, scale_norm):
        super(SublayerConnection, self).__init__()
        if scale_norm=="DyT":
            self.norm=DynamicTanh(atom_dim,True,0.5)
        else:
            self.norm=nn.LayerNorm(atom_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x=x.to("cuda")
        return x + self.dropout(sublayer(self.norm(x)))
class EncoderLayer(nn.Module):
    def __init__(self,h,d_model,dropout_attn,leaky_relu_slope,dropout_feedforward,lambda_attention,trainable_lambda,N_dense,scale_norm):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h,d_model,dropout_attn,lambda_attention,trainable_lambda)
        self.feed_forward=PositionwiseFeedForward(d_model,N_dense,dropout_feedforward,leaky_relu_slope)
        self.d_model=d_model
        self.sublayer=clones(SublayerConnection(d_model,dropout_feedforward,scale_norm).to("cuda"), 2)
    def forward(self,x,mask,adjacency_matrix):
        x=self.sublayer[0](x,lambda x: self.self_attn(x,x,x,mask,adjacency_matrix))
        return self.sublayer[1](x,self.feed_forward)
class Encoder(nn.Module):
    def __init__(self,layer,N,scale_norm,atom_dim):
        super(Encoder, self).__init__()
        self.layers=clones(layer,N)
        if scale_norm == "DyT":
            self.norm = DynamicTanh(atom_dim, True, 0.5)
        else:
            self.norm = nn.LayerNorm(atom_dim)
    def forward(self,x,mask,adjacency_matrix):
        for layer in self.layers:
            x=layer(x,mask,adjacency_matrix)
        return self.norm(x)
class GraphTransformer(nn.Module):
    def __init__(self,d_atom,N,h,d_model,dropout_attn,leaky_relu_slope,dropout_feedforward,lambda_attention,trainable_lambda,N_dense,scale_norm):
        super(GraphTransformer, self).__init__()
        layer=EncoderLayer(h,d_model,dropout_attn,leaky_relu_slope,dropout_feedforward,lambda_attention,trainable_lambda,N_dense,scale_norm)
        self.encoder=Encoder(layer,N,scale_norm,d_model)
        self.src_embed=Embeddings(d_model,d_atom,dropout_feedforward)
    def forward(self,x,mask,adjacency_matrix):
        Transformer_features=self.encoder(self.src_embed(x),mask,adjacency_matrix)
        #Transformer_features:torch.tensor shape[total_atom_number,d_model]
        #all_transformer_layer_features:[N,total_atom_number,d_model]
        #T_0:[total_atom_number,d_model]
        return Transformer_features
class GeLULayerForTransformer(nn.Module):
    def __init__(self,output_features,d_atom,N,h,d_model,dropout_attn,leaky_relu_slope,dropout_feedforward,lambda_attention,trainable_lambda,N_dense,scale_norm):
        super(GeLULayerForTransformer, self).__init__()
        self.linear=nn.Linear(d_model,output_features).to("cuda")
        self.GeLU=nn.GELU().to("cuda")
        self.GraphTransformer=GraphTransformer(d_atom,N,h,d_model,dropout_attn,leaky_relu_slope,dropout_feedforward,lambda_attention,trainable_lambda,N_dense,scale_norm).to("cuda")
    def forward(self,x,mask,adjacency_matrix):
        Transformer_features=self.GraphTransformer(x,mask,adjacency_matrix)
        GeLUTransformer_features=self.GeLU(self.linear(Transformer_features))#[total_atom_number,features_dim]
        return GeLUTransformer_features,Transformer_features
class GeLULayerForGATLayer(nn.Module):
    def __init__(self,in_features,bond_features_dim,output_features,dropout_gnn_ratio,leaky_alpha,elu_alpha,number_layer):
        super(GeLULayerForGATLayer, self).__init__()
        self.linear=nn.Linear(output_features,output_features).to("cuda")
        self.GeLU=nn.GELU()
        self.GATlayerV1=GATlayerV1(in_features,bond_features_dim,output_features,dropout_gnn_ratio,leaky_alpha,elu_alpha).to("cuda")
        self.GATlayerV2=GATlayerV2(output_features,dropout_gnn_ratio,leaky_alpha,elu_alpha).to("cuda")
        self.number_layer=number_layer
    def forward(self,atom_features,bond_features):
        all_tensors=[]
        for i in range(1,self.number_layer+1):
            if i==1:
                GAT_output_features,G_0=self.GATlayerV1(atom_features,bond_features)#[total_atom_number,features_dim]
            else:
                GAT_output_features=self.GATlayerV2(GAT_output_features,bond_features)#[total_atom_number,features_dim]
            all_tensors.append(GAT_output_features)
        all_tensors=torch.stack(all_tensors)#[n,total_atom_number,features_dim]
        averaged_tensor=all_tensors.mean(dim=0)#[total_atom_number,features_dim]
        output=self.GeLU(self.linear(averaged_tensor))
        return output,all_tensors,G_0,averaged_tensor#[total_atom_dim,features_dim]
class AdaptiveMixtureLayer(nn.Module):
    def __init__(self,d_model,GAT_ratio,in_features,bond_features_dim,output_features,dropout_gnn_ratio,leaky_alpha,
                 elu_alpha,N,h,dropout_attn,dropout_feedforward,lambda_attention,trainable_lambda,N_dense,scale_norm,
                 number_layer):
        super(AdaptiveMixtureLayer, self).__init__()
        self.GeLUGAT=GeLULayerForGATLayer(in_features,bond_features_dim,output_features,dropout_gnn_ratio,leaky_alpha,elu_alpha,number_layer)
        self.GeLUTransformer=GeLULayerForTransformer(output_features,in_features,N,h,d_model,dropout_attn,leaky_alpha,dropout_feedforward,lambda_attention,trainable_lambda,N_dense,scale_norm)
        self.GAT_ratio=GAT_ratio
        Transformer_ratio=1.- self.GAT_ratio
        ratio=torch.tensor([Transformer_ratio,GAT_ratio],requires_grad=True)
        self.ratio=torch.nn.Parameter(ratio)
    def forward(self,x,mask,adjacency_matrix,bond_features):
        GATout,all_tensors,G_0,averaged_tensor=self.GeLUGAT(x,bond_features)#[total_atom_numbers,features_dim]
        TransformerOut,Transformer_features=self.GeLUTransformer(x,mask,adjacency_matrix)#[total_atom_numbers,features_dim]
        fused_atom=self.ratio[0]*TransformerOut+self.ratio[1]*GATout
        return fused_atom,all_tensors,G_0,TransformerOut,averaged_tensor,Transformer_features #形状为[total_atom_numbers,features_dim]
class MoleculeAttentionLayer(nn.Module):
    def __init__(self, GAT_ratio, in_features, bond_features_dim, output_features, dropout_gnn_ratio, leaky_alpha,
                 elu_alpha, N, h, dropout_attn, dropout_feedforward,d_model,lambda_attention, trainable_lambda, N_dense,
                 scale_norm, number_layer, molecule_layer_number):
        super(MoleculeAttentionLayer, self).__init__()
        self.AdaptiveMixtureLayer=AdaptiveMixtureLayer(d_model,GAT_ratio,in_features,bond_features_dim,output_features,dropout_gnn_ratio,
                                                       leaky_alpha,elu_alpha,N,h,dropout_attn,dropout_feedforward,lambda_attention,
                                                       trainable_lambda,N_dense,scale_norm,number_layer).to("cuda")
        self.molecule_layer_number=molecule_layer_number
        self.leaky_alpha=leaky_alpha
        self.leakyrelu=nn.LeakyReLU(self.leaky_alpha).to("cuda")
        self.dropout=nn.Dropout(dropout_attn).to("cuda")
        self.molecule_linear1=nn.Linear(output_features*3,output_features).to("cuda")
        self.molecule_linear2=nn.Linear(output_features*2,output_features).to("cuda")
        self.elu_alpha=elu_alpha
        self.GRUcell = nn.GRUCell(output_features,output_features).to("cuda")
    def forward(self,x,mask,adjacency_matrix,bond_features,atom_index):
        new_embedding,all_tensors,G_0,GeLUTransformerOut,averaged_tensor,Transformer_features=self.AdaptiveMixtureLayer(x,mask,adjacency_matrix,bond_features)
        molecule_embedding_list=[]
        molecule_number=len(atom_index)
        for index_number in atom_index:
            initial_index,atom_number=index_number[0],index_number[1]
            single_molecule=new_embedding[initial_index:initial_index+atom_number]
            summed=single_molecule.sum(dim=0)
            molecule_embedding_list.append(summed)
        molecule_stack_tensor=torch.stack(molecule_embedding_list,dim=0)#[molecule_number,molecule_dim]
        for k in range(self.molecule_layer_number+1):
            list_for_molecule = []
            for i in range(molecule_number):
                list_for_molecule_atom3=[]
                list_for_molecule_atom2=[]
                new_index,new_atom_number=atom_index[i][0],atom_index[i][1]
                for j in range(new_index,new_index+new_atom_number):
                    if k==0:
                        three_cat=torch.cat([molecule_stack_tensor[i],G_0[j],GeLUTransformerOut[j]],dim=0)#[3*output_features]
                        two_cat=torch.cat([G_0[j],GeLUTransformerOut[j]],dim=0)
                    else:
                        three_cat=torch.cat([molecule_stack_tensor[i],all_tensors[k-1][i],GeLUTransformerOut[i]],dim=0)
                        two_cat=torch.cat([all_tensors[k-1][i],GeLUTransformerOut[i]],dim=0)
                    list_for_molecule_atom2.append(two_cat)
                    list_for_molecule_atom3.append(three_cat)
                molecule_new1=torch.stack(list_for_molecule_atom3,dim=0)#[single_molecule_atom_number,3*output_features]
                molecule_new2=self.leakyrelu(self.molecule_linear1(self.dropout(molecule_new1)))#[single_molecule_atom_number,output_features]
                molecule_score=F.softmax(molecule_new2,dim=0)#[single_molecule_atom_number,1]
                molecule_new3=self.molecule_linear2(self.dropout(torch.stack(list_for_molecule_atom2,dim=0)))#[single_molecule_atom_number,output_featurs]
                molecule_transform=molecule_score*molecule_new3
                molecule_context=F.elu(torch.sum(molecule_transform,dim=0),alpha=self.elu_alpha)
                list_for_molecule.append(molecule_context)#[output_features]
            list_for_molecule2=torch.stack(list_for_molecule,dim=0)#[molecule_number,output_dim]
            if k==0:
                GRU_results=self.GRUcell(list_for_molecule2,molecule_stack_tensor)
            else:
                GRU_results=self.GRUcell(list_for_molecule2,GRU_results)
        return new_embedding,GRU_results,averaged_tensor,Transformer_features
class Crossattention(nn.Module):
    def __init__(self,is_classif,task_num,fp_Linear_dim,dropout_FPN,cuda,hidden_size,fp_type,hidden_dim,GAT_ratio,in_features,bond_features_dim,output_features,
                 dropout_gnn_ratio,leaky_alpha,elu_alpha,N,h,dropout_attn,dropout_feedforward,d_model,lambda_attention,trainable_lambda,N_dense,
                 scale_norm,number_layer,molecule_layer_number):
        super(Crossattention,self).__init__()
        self.MoleculeAttentionLayer=MoleculeAttentionLayer(GAT_ratio,in_features,bond_features_dim,output_features,
                                                           dropout_gnn_ratio,leaky_alpha,elu_alpha,N,h,dropout_attn,dropout_feedforward,d_model,
                                                           lambda_attention,trainable_lambda,N_dense,scale_norm,number_layer,molecule_layer_number).to("cuda")
        self.FPN=FPN(fp_Linear_dim,dropout_FPN,cuda,hidden_size,fp_type).to("cuda")
        self.to_query=nn.Linear(hidden_size,hidden_dim).to("cuda")
        self.to_key=nn.Linear(output_features,hidden_dim).to("cuda")
        self.to_value=nn.Linear(output_features,hidden_dim).to("cuda")
        self.to_out=nn.Linear(hidden_dim,output_features).to("cuda")
        self.is_classif=is_classif
        self.ffn=nn.Sequential(
            nn.Dropout(dropout_feedforward).to("cuda"),
            nn.Linear(in_features=output_features,out_features=output_features,bias=True).to("cuda"),
            nn.ReLU(),
            nn.Dropout(dropout_feedforward).to("cuda"),
            nn.Linear(in_features=output_features,out_features=task_num,bias=True).to("cuda"),)
        self.sigmoid=nn.Sigmoid()
    def forward(self,smiles,atom_features,mask,adjacency_matrix,bond_features,atom_index):
        FPN_output=self.FPN(smiles)
        new_embeddings,Graph_output,averaged_tensor,Transformer_features=self.MoleculeAttentionLayer(atom_features,mask,adjacency_matrix,bond_features,atom_index)#[分子数,output_features]
        Q,K,V=self.to_query(FPN_output),self.to_key(Graph_output),self.to_value(Graph_output)
        attn_scores=torch.matmul(Q,K.transpose(-2,-1))
        attn_scores = attn_scores / (K.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        output = self.to_out(attn_output)
        output=self.ffn(output)
        if self.is_classif and not self.training:
            output=self.sigmoid(output)
        return output,FPN_output,Graph_output,averaged_tensor,Transformer_features,new_embeddings
def MultiMixedInformationNetwork(dataset_type,task_num,fp_Linear_dim,dropout_FPN,cuda,hidden_size,fp_type,hidden_dim,GAT_ratio,in_features,bond_features_dim,output_features,
                 dropout_gnn_ratio,leaky_alpha,elu_alpha,N,h,dropout_attn,dropout_feedforward,d_model,lambda_attention,trainable_lambda,N_dense,
                 scale_norm,number_layer,molecule_layer_number):
    if dataset_type=="classification":
        is_classif=1
    else:
        is_classif=0
    model=Crossattention(is_classif,task_num,fp_Linear_dim,dropout_FPN,cuda,hidden_size,fp_type,hidden_dim,GAT_ratio,in_features,bond_features_dim,output_features,
                 dropout_gnn_ratio,leaky_alpha,elu_alpha,N,h,dropout_attn,dropout_feedforward,d_model,lambda_attention,trainable_lambda,N_dense,
                 scale_norm,number_layer,molecule_layer_number)
    for param in model.parameters():
        if param.dim()==1:
            nn.init.constant_(param,0)
        else:
            nn.init.kaiming_uniform_(param, nonlinearity='leaky_relu', a=0.01)
    return model