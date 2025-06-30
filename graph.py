import numpy as np
from rdkit import Chem
import torch
import pandas as pd
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import Dataset
import pickle
def one_of_k_encoding(x, allowable_set):
    l=[]
    for i in allowable_set:
        if x==i:
            l+=[1]
        else:
            l+=[0]
    return l
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    l=[]
    if x not in allowable_set:
        x = allowable_set[-1]
    for i in allowable_set:
        if x==i:
            l+=[1]
        else:
            l+=[0]
    return l
def get_atom_features(atom,mol):
    hydrogen_donor=Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
    hydrogen_acceptor = Chem.MolFromSmarts("[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
        "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
    acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
    basic = Chem.MolFromSmarts("[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
        "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")
    hydrogen_donor_match=sum(mol.GetSubstructMatches(hydrogen_donor), ())
    hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
    acidic_match = sum(mol.GetSubstructMatches(acidic), ())
    basic_match = sum(mol.GetSubstructMatches(basic), ())
    atom_idx=atom.GetIdx()
    ring_info = mol.GetRingInfo()
    attributes=[]
    attributes+=one_of_k_encoding_unk(atom.GetSymbol(),['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','other'])
    attributes+=one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5])
    attributes+=[atom.GetFormalCharge(),atom.GetNumRadicalElectrons()]
    attributes+=one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,'other'])
    attributes+=[1 if atom.GetIsAromatic() else 0]
    attributes+=one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])
    attributes+=[1 if atom.IsInRing() else 0]
    attributes=(attributes+[1 if ring_info.IsAtomInRingOfSize(atom_idx, 3) else 0]+[1 if ring_info.IsAtomInRingOfSize(atom_idx, 4) else 0]
                +[1 if ring_info.IsAtomInRingOfSize(atom_idx, 5) else 0]+[1 if ring_info.IsAtomInRingOfSize(atom_idx, 6) else 0])
    attributes+=one_of_k_encoding(int(atom.GetChiralTag()),[0, 1, 2, 3])
    attributes+=[atom.GetMass()*0.01]#
    attributes+=one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
    attributes=(attributes+[1 if atom_idx in hydrogen_acceptor_match else 0] +
                [1 if atom_idx in hydrogen_donor_match else 0] + [1 if atom_idx in acidic_match else 0] + [1 if atom_idx in basic_match else 0])#4维
    return attributes
bond_fdim=13
def get_bond_features(bond):
    if bond is None:
        fbond=[1]+[0]*(bond_fdim-1)
    else:
        bt=bond.GetBondType()
        fbond= [
                0,
                1 if bt == Chem.rdchem.BondType.SINGLE else 0,
                1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
                1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
                1 if bt == Chem.rdchem.BondType.AROMATIC else 0,
                1 if bond.GetIsConjugated()  else 0,
                1 if bond.IsInRing()  else 0
            ]
        fbond += one_of_k_encoding(int(bond.GetStereo()), list(range(6)))
    return fbond#键特征总共13维
def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    mol = Chem.MolFromSmiles('CC')
    alist = mol.GetAtoms()
    a = alist[0]
    return len(get_atom_features(a,mol))#57维
def num_bond_features():
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(get_bond_features(simple_mol.GetBonds()[0]))
class GraphOne:
    def __init__(self,smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(smiles)
        self.smiles=smiles
        self.atom_features=[]
        self.bond_features={}
        self.atom_numbers=mol.GetNumAtoms()
        self.bond_numbers=mol.GetNumBonds()
        self.edge_index=[]
        for i,atom in enumerate(mol.GetAtoms()):
            self.atom_features.append(get_atom_features(atom,mol))
        self.atom_features=[self.atom_features[i] for i in range(self.atom_numbers)]
        self.adjacency_matrix=torch.eye(self.atom_numbers)
        for i,bond in enumerate(mol.GetBonds()):
            begin_atom,end_atom=bond.GetBeginAtom().GetIdx(),bond.GetEndAtom().GetIdx()
            self.edge_index+=[(begin_atom,end_atom),(end_atom,begin_atom)]
            self.bond_features[(begin_atom,end_atom)]=get_bond_features(bond)
            self.bond_features[(end_atom,begin_atom)]=get_bond_features(bond)
            self.adjacency_matrix[begin_atom,end_atom]=1
            self.adjacency_matrix[end_atom,begin_atom]=1
class GraphBatch:
    def __init__(self,graphs):
        smile_list=[]
        for graph in graphs:
            smile_list.append(graph)
        self.smile_list=smile_list
        self.smile_num=len(self.smile_list)
        self.atom_features_dim=num_atom_features()
        self.bond_features_dim=num_bond_features()
        self.atom_no=0
        self.atom_index=[]
        self.bond_features={}
        atom_features=[]
        for graph in graphs:
            atom_features.extend(graph.atom_features)
            self.atom_index.append((self.atom_no,graph.atom_numbers))
            for index,value in graph.bond_features.items():
                begin=index[0]+self.atom_no
                end=index[1]+self.atom_no
                self.bond_features[(begin,end)]=torch.FloatTensor(value)
                self.bond_features[(end,begin)]=torch.FloatTensor(value)
            self.atom_no = self.atom_no + graph.atom_numbers
        self.atom_features=torch.FloatTensor(atom_features)
        self.adjacency_matrix=torch.eye(self.atom_no)
        for i in self.bond_features.keys():
            begin_atom,end_atom=i[0],i[1]
            self.adjacency_matrix[begin_atom,end_atom]=1
            self.adjacency_matrix[end_atom,begin_atom]=1
    def get_feature(self):
        return self.atom_features,self.atom_index,self.bond_features
    def get_adjacency_matrix(self):
        return self.adjacency_matrix
def create_graph(smiles):
    graphs=[]
    for one in smiles:
        graph=GraphOne(one)
        graphs.append(graph)
    return GraphBatch(graphs)