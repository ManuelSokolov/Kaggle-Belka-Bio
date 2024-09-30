import dask.dataframe as dd
import pandas as pd
import numpy as np 
import random

dataset_path = 'dataset/train.parquet'

# Load the dataset using Dask
train = dd.read_parquet(dataset_path)

bblocks = set(train.buildingblock1_smiles)

bblocks2 = set(train.buildingblock2_smiles)

bblocks3 = set(train.buildingblock3_smiles)

shared_blocks = bblocks & (bblocks3 | bblocks2)

bblocksall = set(bblocks | bblocks2 | bblocks3)
bbdict = {x:i for i,x in enumerate(bblocksall)}

#pickle.dump(bbdict, open('bbdict.pickle', 'bw'))

print("pickle built")


def get_encoded(dataset_path, col, BBs_dict):
    BBs = pd.read_parquet(dataset_path, engine = 'pyarrow', columns=[col])
    BBs = BBs[col].to_numpy()
    #BBs_reshaped = np.reshape(BBs, [-1, 3])
    #BBs = BBs_reshaped[:, 0]
    encoded_BBs = [BBs_dict[x] for x in BBs]
    encoded_BBs = np.asarray(encoded_BBs, dtype = np.int16)
    return encoded_BBs

print("Obtaining BB encodings")
encoded_BBs_1 = get_encoded(dataset_path, 'buildingblock1_smiles', bbdict)
encoded_BBs_2 = get_encoded(dataset_path, 'buildingblock2_smiles', bbdict)
encoded_BBs_3 = get_encoded(dataset_path, 'buildingblock3_smiles', bbdict)


def get_molecule_smiles(dataset_path):
    molecule_smiles = pd.read_parquet(dataset_path, engine = 'pyarrow', columns=['molecule_smiles'])
    molecule_smiles = molecule_smiles.molecule_smiles.to_numpy()
    #molecule_smiles = np.reshape(molecule_smiles, [-1, 3])
    #if np.mean(molecule_smiles[:, 0] == molecule_smiles[:, 1]) != 1:
    #    print('ERROR')
    #if np.mean(molecule_smiles[:, 0] == molecule_smiles[:, 2]) != 1:
    #    print('ERROR')
    return molecule_smiles

molecule_smiles = get_molecule_smiles(dataset_path)

print("molecule smiles obtained")

def get_binds(dataset_path):
    binds =  pd.read_parquet(dataset_path, engine = 'pyarrow', columns=['binds']).binds.to_numpy()
    #return np.reshape(binds.astype('byte'), [-1, 3])
    return binds

binds = get_binds(dataset_path)
protein_name = pd.read_parquet(dataset_path, engine = 'pyarrow', columns=['protein_name']).protein_name.to_numpy()
id_var = pd.read_parquet(dataset_path, engine = 'pyarrow', columns=['id']).id.to_numpy()

print("protein read")

# Check lengths of all arrays
print(f"Length of encoded_BBs_1: {len(encoded_BBs_1)}")
print(f"Length of encoded_BBs_2: {len(encoded_BBs_2)}")
print(f"Length of encoded_BBs_3: {len(encoded_BBs_3)}")
print(f"Length of molecule_smiles: {len(molecule_smiles)}")
print(f"Length of binds: {len(binds)}")
print(f"Length of protein_name: {len(protein_name)}")


data = {'buildingblock1_smiles':encoded_BBs_1, 'buildingblock2_smiles':encoded_BBs_2, 'buildingblock3_smiles':encoded_BBs_3,
        'molecule_smiles':molecule_smiles, 'binds':binds,'protein_name': protein_name,'id': id_var}

df = pd.DataFrame(data=data)

random.seed(42)
bb1ids = df.buildingblock1_smiles.unique().tolist()
bb2ids = df.buildingblock2_smiles.unique().tolist()
bb3ids = df.buildingblock3_smiles.unique().tolist()

group2 = list(set(bb2ids) & set(bb3ids))
group3 = list(set(bb3ids) - set(bb2ids))

bbs1 = random.sample(bb1ids, 17)
bbs2 = random.sample(group2, 34)
bbs3 = random.sample(group3, 2)

df['bb_test_noshare'] = False

df['bb_test_noshare'].iloc[df.buildingblock1_smiles.isin(bbs1) & df.buildingblock2_smiles.isin(bbs2) & df.buildingblock3_smiles.isin(bbs2+bbs3)] = True

df['bb_test_mixshare'] = False

df['bb_test_mixshare'].iloc[df.buildingblock1_smiles.isin(bbs1) | \
df.buildingblock2_smiles.isin(bbs2) | df.buildingblock3_smiles.isin(bbs2+bbs3)] \
= True

df['random_test'] = False

random.seed(42)
random_indices = random.sample(range(len(df)), int(len(df)*0.003))

df['random_test'].iloc[random_indices] = True

df[['bb_test_noshare', 'bb_test_mixshare', 'random_test']].mean()

df['full_test'] = df['bb_test_noshare'] + df['bb_test_mixshare'] + df['random_test']
df['full_test'] = df['full_test'] > 0

df.to_parquet('train_folds.parquet', index=False)

