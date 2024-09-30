# Opening Dataset
import pandas as pd
import duckdb
from tqdm import tqdm
import concurrent.futures
# Featurization
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as L
import polars as pl
from tqdm import tqdm
from torchmetrics import AveragePrecision, Precision, Recall, AUROC

import numpy as np


con = duckdb.connect()

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding
    
    
# Function to find the longest common substring
# Feat Salty
def longest_common_substring(str1, str2):
    m = [[0] * (1 + len(str2)) for i in range(1 + len(str1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(str1)):
        for y in range(1, 1 + len(str2)):
            if str1[x - 1] == str2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return str1[x_longest - longest: x_longest], x_longest - longest

    
# Main atom feat. func

def get_atom_features(atom, position_in_smile_bb2, len_common_substring2, position_in_smile_bb3, len_common_substring3, current_index, use_chirality=True):
    # Define a simplified list of atom types
    permitted_atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Dy', 'Unknown']
    atom_type = atom.GetSymbol() if atom.GetSymbol() in permitted_atom_types else 'Unknown'
    atom_type_enc = one_hot_encoding(atom_type, permitted_atom_types)
    
    # Consider only the most impactful features: atom degree and whether the atom is in a ring
    atom_degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 'MoreThanFour'])
    is_in_ring = [int(atom.IsInRing())]
    
    explicit_valence = [atom.GetExplicitValence()]
    implicit_valence = [atom.GetImplicitValence()]
    formal_charge = [atom.GetFormalCharge()]
    num_radical_electrons = [atom.GetNumRadicalElectrons()]
    
    is_in_bb1 = [int(position_in_smile_bb2 <= current_index < position_in_smile_bb2 + len_common_substring2)]
    is_in_bb2 = [int(position_in_smile_bb3 <= current_index < position_in_smile_bb3 + len_common_substring3)]
    
    # Optionally include chirality
    if use_chirality:
        chirality_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_features = atom_type_enc + atom_degree + is_in_ring + chirality_enc + implicit_valence + explicit_valence + formal_charge + num_radical_electrons + is_in_bb1 + is_in_bb2
    else:
        atom_features = atom_type_enc + atom_degree + is_in_ring + is_in_bb1 + is_in_bb2
    
    return np.array(atom_features, dtype=np.float32)

def get_bond_features(bond):
    # Simplified list of bond types
    permitted_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC, 'Unknown']
    bond_type = bond.GetBondType() if bond.GetBondType() in permitted_bond_types else 'Unknown'
    bond_angle = [bond.GetBondAngle()] if bond.HasProp("BondAngle") else [0.0]
    
    # Features: Bond type, Is in a ring
    features = one_hot_encoding(bond_type, permitted_bond_types) \
               + [int(bond.IsInRing())] 
    
    return np.array(features, dtype=np.float32)



class DatasetLoader(Dataset):
    """Dataset for fine-tuning Molformer on ligand binding dataset."""
    def __init__(
        self,
        dataset_file: str,
        stage: str = "train",
        protein: str = "sEH",
        positives: bool = False,
        number: int = 0 
    ):
        self.stage = stage
        #df = pl.scan_parquet(dataset_file)
        if stage == "test":
            columns = ["id", "molecule_smiles", "buildingblock2_smiles", "buildingblock3_smiles"]
            columns_str = ', '.join(columns)
            query = f"""
            SELECT {columns_str} FROM parquet_scan('{dataset_file}')
            WHERE protein_name = '{protein}'
            """
        elif stage == "val":
            columns = ["id", "molecule_smiles", "binds", "buildingblock2_smiles", "buildingblock3_smiles"]
            columns_str = ', '.join(columns)
            query = f"""
            SELECT {columns_str} FROM parquet_scan('{dataset_file}')
            WHERE protein_name = '{protein}'
            """
        else:
            columns = ["id", "molecule_smiles", "binds","buildingblock2_smiles", "buildingblock3_smiles"]
            columns_str = ', '.join(columns)
            if not positives:
                query = f"""
                SELECT {columns_str} FROM parquet_scan('{dataset_file}')
                WHERE protein_name = '{protein}'
                """
            else:     
                query = f"""
                (SELECT * FROM parquet_scan('{dataset_file}')
                WHERE binds = 0 AND protein_name = '{protein}'
                ORDER BY random()
                LIMIT {number})
                UNION ALL
                (SELECT * FROM parquet_scan('{dataset_file}')
                 WHERE binds = 1 AND protein_name = '{protein}'
                 ORDER BY random()
                 LIMIT {number})
                 """      
        self.dataset = con.query(query).df()
                    
        #self.dataset = df.select(columns).fetch(100)
        
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)
    def __getitem__(self, idx: int):
        """Return the sample at the given index."""

        id_molecule = self.dataset.iloc[idx]["id"]
        smiles = str(self.dataset.iloc[idx]["molecule_smiles"])
        bb2 = str(self.dataset.iloc[idx]["buildingblock2_smiles"])
        bb3 = str(self.dataset.iloc[idx]["buildingblock3_smiles"])
        
        common_sub_string2 , position_in_smile_bb2 = longest_common_substring(smiles, bb2)
        len_common_substring2 = len(common_sub_string2)
        common_sub_string3 , position_in_smile_bb3 = longest_common_substring(smiles, bb3)
        len_common_substring3 = len(common_sub_string3)

        if self.stage == "test":
            label = np.array(-1)
        else:
            label = self.dataset.iloc[idx]["binds"]

        mol = Chem.MolFromSmiles(smiles)

        # Node features
        atom_features = [get_atom_features(atom, position_in_smile_bb2, len_common_substring2, position_in_smile_bb3, len_common_substring3, idx) for idx, atom in enumerate(mol.GetAtoms())]
        x = torch.tensor(np.vstack(atom_features), dtype=torch.float)  # Convert list of numpy arrays to a single numpy array before torch.tensor

        # Edge features
        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index += [(start, end), (end, start)]  # Undirected graph
            bond_feature = get_bond_features(bond)
            edge_features += [bond_feature, bond_feature]  # Same features in both directions

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)  # Convert list of numpy arrays to a single numpy array before torch.tensor

        # Creating the Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.molecule_id = id_molecule
        data.y = torch.tensor(label, dtype=torch.float)

        return data
    

    def get_number_of_positives(self):
        """Return the number of positive samples in the dataset where 'binds' equals 1."""
        if 'binds' in self.dataset.columns:
            return self.dataset['binds'].sum()
        else:
            raise ValueError("Dataset does not contain 'binds' column.")

