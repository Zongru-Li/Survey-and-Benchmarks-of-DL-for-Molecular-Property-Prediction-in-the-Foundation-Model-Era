import dgl
import torch
import numpy as np
import math

from rdkit import Chem
from rdkit.Chem import AllChem
from jarvis.core.specie import chem_data, get_node_attributes


def calculate_dis(A, B):
    AB = B - A
    dis = np.linalg.norm(AB)
    return dis


def encode_chirality(atom):
    chirality_tags = [0] * 4
    if atom.HasProp("_CIPCode"):
        chirality = atom.GetProp("_CIPCode")
        if chirality == "R":
            chirality_tags[0] = 1
        elif chirality == "S":
            chirality_tags[1] = 1
        elif chirality == "E":
            chirality_tags[2] = 1
        elif chirality == "Z":
            chirality_tags[3] = 1
    return chirality_tags


def bond_length_approximation(bond_type):
    bond_length_dict = {"SINGLE": 1.0, "DOUBLE": 1.4, "TRIPLE": 1.8, "AROMATIC": 1.5}
    return bond_length_dict.get(bond_type, 1.0)


def get_bond_formal_charge(bond):
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    chirality_tags_atom = encode_chirality(atom1)
    chirality_tags_atom.extend(encode_chirality(atom2))
    return chirality_tags_atom


def bond_type_map(bond_type):
    bond_length_dict = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
    return bond_length_dict[bond_type]


def bond_stereo_onehot(bond):
    stereo = bond.GetStereo()
    if stereo == Chem.BondStereo.STEREOANY:
        return [1, 0, 0, 0, 0]
    elif stereo == Chem.BondStereo.STEREOCIS:
        return [0, 1, 0, 0, 0]
    elif stereo == Chem.BondStereo.STEREOTRANS:
        return [0, 0, 1, 0, 0]
    elif stereo == Chem.BondStereo.STEREONONE:
        return [0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 1]


def bond_length_onehot(bond_length):
    if 1.2 <= bond_length <= 1.6:
        return [1, 0, 0, 0, 0]
    elif 1.3 <= bond_length <= 1.5:
        return [0, 1, 0, 0, 0]
    elif 1.1 <= bond_length <= 1.3:
        return [0, 0, 1, 0, 0]
    elif 1.4 <= bond_length <= 1.5:
        return [0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 1]


def encode_bond_26(bond, mol):
    bond_dir = [0] * 7
    bond_dir[bond.GetBondDir()] = 1
    bond_type = [0] * 4
    bond_type[bond_type_map(str(bond.GetBondType()))] = 1
    bond_lg = AllChem.GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    bond_length = bond_length_onehot(bond_lg)
    in_ring = [0] * 2
    in_ring[int(bond.IsInRing())] = 1
    stereo = get_bond_formal_charge(bond)
    return bond_dir + bond_type + bond_length + in_ring + stereo


def encode_bond_14(bond):
    bond_dir = [0] * 7
    bond_dir[bond.GetBondDir()] = 1
    bond_type = [0] * 4
    bond_type[int(bond.GetBondTypeAsDouble()) - 1] = 1
    bond_length = bond_length_approximation(bond.GetBondType())
    in_ring = [0, 0]
    in_ring[int(bond.IsInRing())] = 1
    non_bond_feature = [0] * 6
    edge_encode = bond_dir + bond_type + [bond_length, bond_length**2] + in_ring + non_bond_feature
    return edge_encode


def non_bonded(charge_list, i, j, dis):
    charge_list = [float(charge) for charge in charge_list]
    q_i = [charge_list[i]]
    q_j = [charge_list[j]]
    q_ij = [charge_list[i] * charge_list[j]]
    dis_1 = [1 / dis]
    dis_2 = [1 / (dis**6)]
    dis_3 = [1 / (dis**12)]
    return q_i + q_j + q_ij + dis_1 + dis_2 + dis_3


def mmff_force_field(mol):
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
        return True
    except ValueError:
        return False


def uff_force_field(mol):
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFGetMoleculeForceField(mol)
        return True
    except ValueError:
        return False


def random_force_field(mol):
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42)
        return True
    except ValueError:
        return False


def check_common_elements(list1, list2, element1, element2):
    for i in range(len(list1)):
        if list1[i] == element1 and list2[i] == element2:
            return True
    return False


def tensor_nan_inf(per_bond_feat):
    nan_exists = any(math.isnan(x) if isinstance(x, float) else False for x in per_bond_feat)
    inf_exists = any(x == float('inf') if isinstance(x, float) else False for x in per_bond_feat)
    ninf_exists = any(x == float('-inf') if isinstance(x, float) else False for x in per_bond_feat)
    if nan_exists or inf_exists or ninf_exists:
        clean_list = [0 if isinstance(x, float) and math.isnan(x) else x for x in per_bond_feat]
        per_bond_feat = [1 if x == float('inf') else -1 if x == float('-inf') else x for x in clean_list]
        return per_bond_feat
    else:
        return per_bond_feat


def atom_to_graph(smiles, encoder_atom, encoder_bond):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    else:
        mol = Chem.AddHs(mol)
    
    sps_features = []
    coor = []
    edge_id = []
    atom_charges = []
    
    smiles_with_hydrogens = Chem.MolToSmiles(mol)
    tmp = []
    for num in smiles_with_hydrogens:
        if num not in ['[', ']', '(', ')']:
            tmp.append(num)
    
    sm = {}
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        sm[atom_index] = atom.GetSymbol()
    
    Num_toms = len(tmp)
    if Num_toms > 700:
        return False
    
    if mmff_force_field(mol) == True:
        num_conformers = mol.GetNumConformers()
        if num_conformers > 0:
            AllChem.ComputeGasteigerCharges(mol)
            for ii, s in enumerate(mol.GetAtoms()):
                per_atom_feat = []
                feat = list(get_node_attributes(s.GetSymbol(), atom_features=encoder_atom))
                per_atom_feat.extend(feat)
                sps_features.append(per_atom_feat)
                pos = mol.GetConformer().GetAtomPosition(ii)
                coor.append([pos.x, pos.y, pos.z])
                charge = s.GetProp("_GasteigerCharge")
                atom_charges.append(charge)
            
            edge_features = []
            src_list, dst_list = [], []
            for bond in mol.GetBonds():
                bond_type = bond.GetBondTypeAsDouble()
                src = bond.GetBeginAtomIdx()
                dst = bond.GetEndAtomIdx()
                src_list.append(src)
                src_list.append(dst)
                dst_list.append(dst)
                dst_list.append(src)
                src_coor = np.array(coor[src])
                dst_coor = np.array(coor[dst])
                s_d_dis = calculate_dis(src_coor, dst_coor)
                per_bond_feat = []
                per_bond_feat.extend(encode_bond_14(bond))
                edge_features.append(per_bond_feat)
                edge_features.append(per_bond_feat)
                edge_id.append([1])
                edge_id.append([1])
            
            for i in range(len(coor)):
                coor_i = np.array(coor[i])
                for j in range(i + 1, len(coor)):
                    coor_j = np.array(coor[j])
                    s_d_dis = calculate_dis(coor_i, coor_j)
                    if 0 < s_d_dis <= 5:
                        if check_common_elements(src_list, dst_list, i, j):
                            src_list.extend([i, j])
                            dst_list.extend([j, i])
                            per_bond_feat = [0] * 15
                            per_bond_feat.extend(non_bonded(atom_charges, i, j, s_d_dis))
                            clean_list = tensor_nan_inf(per_bond_feat)
                            edge_features.append(clean_list)
                            edge_features.append(clean_list)
                            edge_id.append([0])
                            edge_id.append([0])
            
            coor_tensor = torch.tensor(coor, dtype=torch.float32)
            edge_feats = torch.tensor(edge_features, dtype=torch.float32)
            edge_id_feats = torch.tensor(edge_id, dtype=torch.float32)
            node_feats = torch.tensor(sps_features, dtype=torch.float32)
            num_atoms = mol.GetNumAtoms()
            g = dgl.graph((src_list, dst_list), num_nodes=num_atoms)
            g.ndata['feat'] = node_feats
            g.ndata['coor'] = coor_tensor
            g.edata['feat'] = edge_feats
            g.edata['id'] = edge_id_feats
            return g
        else:
            return False
    else:
        return False


def path_complex_mol(smiles, encoder_atom, encoder_bond):
    g = atom_to_graph(smiles, encoder_atom, encoder_bond)
    if g != False:
        return g
    else:
        return False
