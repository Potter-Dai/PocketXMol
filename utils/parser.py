import os
import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from utils.fragment import find_rotatable_bond_mat
# from rdkit.Chem.rdchem import BondType
# from rdkit.Chem import ChemicalFeatures
# from rdkit import RDConfig

# from utils.mol2frag import mol2frag

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
# BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
# BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}


class PDBLigand(object):

    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto', removeHs=True):
        super().__init__()
        # # Read PDB file
        self.removeHs = removeHs
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if (line[0:6].strip() == 'ATOM') or (line[0:6].strip() == 'HETATM'):
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': line[0:6].strip(),  # ATOM or HETATM
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break   # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            if self.removeHs and atom['element_symb'] == 'H':
                continue
            
            nonstd = (atom['res_name'] not in self.AA_NAME_NUMBER)
            
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                    'chain_res_id': chain_res_id
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass
        
        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name   # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.int64),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.int64),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, mol, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        mol_pos = mol.GetConformer().GetPositions()
        for center in mol_pos:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected
    
    def get_pocket_info(self, selected):
        num_res = len(selected)
        num_atoms = sum([len(residue['atoms']) for residue in selected])
        res_ids = [residue['chain_res_id'] for residue in selected]
        cover_chains = sorted(list(set([residue['chain'] for residue in selected])))
        pocket_info = {
            'num_res': num_res,
            'num_atoms': num_atoms,
            'res_ids': ';'.join(res_ids),
            'num_chains': len(cover_chains),
            'cover_chain_ids': ';'.join(cover_chains),
        }
        return pocket_info

    def get_chain_seqs(self, chains):
        if isinstance(chains, str):
            chains = [chains]
        seqs = []
        for chain in chains:
            seq = []
            for residue in self.residues:
                if residue['chain'] == chain:
                    seq.append(self.AA_NAME_SYM[residue['name']])
            seqs.append(''.join(seq))
            
        return ';'.join(seqs)


    def residues_to_pdb_block(self, residues, name='POCKET'):
        block =  "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block
    
    
def parse_mol_with_confs(mol, smiles=None, confs=None):
    if smiles is not None: # check smiles
        if smiles != Chem.MolToSmiles(mol, isomericSmiles=False):
            return None
    pos_all_confs = []
    i_conf_list = []

    data = parse_3d_mol(mol, smiles=smiles, not_pos=True)
    if data is None:
        return None
    
    # get all confs
    if confs is None:
        for i_conf in range(mol.GetNumConformers()):
            pos = mol.GetConformer(i_conf).GetPositions()
            pos_all_confs.append(pos)
            i_conf_list.append(i_conf)
            # check bond length
            # bond_lengths = np.linalg.norm(pos[data['bond_index'][0]] - pos[data['bond_index'][1]], ord=2, axis=-1)
            # if (bond_lengths > 3.5).any():
            #     print('Skipping conformer with too long bond length: %s' % bond_lengths.max())
            #     return None
    else:
        pos_all_confs = np.array(confs)[:, :len(data['element']), :]
        i_conf_list = list(range(len(confs)))

    return {
        'element': np.array(data['element']),
        'bond_index': np.array(data['bond_index']),
        'bond_type': np.array(data['bond_type']),
        'pos_all_confs': np.array(pos_all_confs, dtype=np.float32),
        'num_atoms': data['num_atoms'],
        'num_bonds': data['num_bonds'],
        'i_conf_list': i_conf_list,
        'num_confs': len(i_conf_list),
    }

class PDBProtein(object):

    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto', removeHs=True):
        super().__init__()
        self.removeHs = removeHs
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break   # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            if self.removeHs and atom['element_symb'] == 'H':
                continue
            if atom['res_name'] not in self.AA_NAME_NUMBER:
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                    'chain_res_id': chain_res_id
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass
        
        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name   # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.int64),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.int64),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, mol, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        mol_pos = mol.GetConformer().GetPositions()
        for center in mol_pos:
            for i, residue in enumerate(self.residues):
                if criterion == 'min':
                    res_pos = np.array([self.pos[atom] for atom in residue['atoms']])
                    distance_all = np.linalg.norm(res_pos - center, ord=2, axis=-1)
                    distance = distance_all.min()
                else:
                    distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected
    
    def get_pocket_info(self, selected):
        num_res = len(selected)
        num_atoms = sum([len(residue['atoms']) for residue in selected])
        res_ids = [residue['chain_res_id'] for residue in selected]
        cover_chains = sorted(list(set([residue['chain'] for residue in selected])))
        pocket_info = {
            'num_res': num_res,
            'num_atoms': num_atoms,
            'res_ids': ';'.join(res_ids),
            'num_chains': len(cover_chains),
            'cover_chain_ids': ';'.join(cover_chains),
        }
        return pocket_info

    def get_chain_seqs(self, chains):
        if isinstance(chains, str):
            chains = [chains]
        seqs = []
        for chain in chains:
            seq = []
            for residue in self.residues:
                if residue['chain'] == chain:
                    seq.append(self.AA_NAME_SYM[residue['name']])
            seqs.append(''.join(seq))
            
        return ';'.join(seqs)


    def residues_to_pdb_block(self, residues, name='POCKET'):
        block =  "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block
    
    
def parse_pdb_peptide(pdb_path):
    parser = PDBParser(PERMISSIVE=0)
    structure = parser.get_structure('structure', pdb_path)
    
    pep_feature_dict = {'pos': [], 'atom_name': [], 'is_backbone': [], 'res_id': [], 'atom_to_aa_type': []}
    for atom in structure.get_atoms():
        element = atom.element
        if element == 'H':
            continue
        element = element if len(element) == 1 else element[0] + element[1].lower()
        pep_feature_dict['pos'].append(atom.get_coord())
        pep_feature_dict['atom_name'].append(atom.get_name())
        res = atom.get_parent()
        pep_feature_dict['res_id'].append(res.get_id()[1])
        resname = res.get_resname()
        aa_type = PDBProtein.AA_NAME_NUMBER[resname] if resname in PDBProtein.AA_NAME_NUMBER else len(PDBProtein.AA_NAME_NUMBER)
        pep_feature_dict['atom_to_aa_type'].append(aa_type)
        pep_feature_dict['is_backbone'].append(
            (atom.get_name() in PDBProtein.BACKBONE_NAMES) and (aa_type < len(PDBProtein.AA_NAME_NUMBER)))

    unique_res_id, res_index = np.unique(pep_feature_dict['res_id'], return_inverse=True)
    pep_feature_dict['res_index'] = res_index
    assert (np.diff(unique_res_id) == 1).all(), 'residue id is not continuous'

    pep_feature_dict = {k: np.array(v) for k, v in pep_feature_dict.items()}
    
    pep_feature_dict['seq'] = ''.join([seq1(residue.get_resname()) or 'X' for residue in structure.get_residues()])
    pep_feature_dict['pep_len'] = len(pep_feature_dict['seq'])
    pep_feature_dict['pep_path'] = pdb_path
    pep_feature_dict['atom_name'] = list(pep_feature_dict['atom_name'])
    
    if 'X' in pep_feature_dict['seq']:
        print('Warning: X in peptide sequence:', pdb_path)
    
    return pep_feature_dict


def parse_mol_with_confs(mol, smiles=None, confs=None):
    if smiles is not None: # check smiles
        if smiles != Chem.MolToSmiles(mol, isomericSmiles=False):
            return None
    pos_all_confs = []
    i_conf_list = []

    data = parse_3d_mol(mol, smiles=smiles, not_pos=True)
    if data is None:
        return None
    
    # get all confs
    if confs is None:
        for i_conf in range(mol.GetNumConformers()):
            pos = mol.GetConformer(i_conf).GetPositions()
            pos_all_confs.append(pos)
            i_conf_list.append(i_conf)
            # check bond length
            # bond_lengths = np.linalg.norm(pos[data['bond_index'][0]] - pos[data['bond_index'][1]], ord=2, axis=-1)
            # if (bond_lengths > 3.5).any():
            #     print('Skipping conformer with too long bond length: %s' % bond_lengths.max())
            #     return None
    else:
        pos_all_confs = np.array(confs)[:, :len(data['element']), :]
        i_conf_list = list(range(len(confs)))

    return {
        'element': np.array(data['element']),
        'bond_index': np.array(data['bond_index']),
        'bond_type': np.array(data['bond_type']),
        'pos_all_confs': np.array(pos_all_confs, dtype=np.float32),
        'num_atoms': data['num_atoms'],
        'num_bonds': data['num_bonds'],
        'i_conf_list': i_conf_list,
        'num_confs': len(i_conf_list),
    }


def parse_conf_list(conf_list, smiles=None):
    # data_list = [parse_drug3d_mol(conf) for conf in conf_list]
    
    element = []
    bond_index = []
    bond_type = []
    pos_all_confs = []
    i_conf_list = []
    num_atoms = 0
    num_bonds = 0  # NOTE: the num of bonds is not symtric
    for i_conf,  conf in enumerate(conf_list):
        data = parse_3d_mol(conf, smiles=smiles)
        if data is None:
            continue
        # check element
        if len(element) == 0:
            element = data['element']
            num_atoms = data['num_atoms']
        else:
            if data['num_atoms'] != num_atoms:
                print('Skipping conformer with different number of atoms')
                continue
            if not np.all(element == data['element']):
                print('Skipping conformer with different element order')
                continue
        # check bond
        if len(bond_index) == 0:
            bond_index = data['bond_index']
            bond_type = data['bond_type']
            num_bonds = data['num_bonds']
        else:
            if data['num_bonds'] != num_bonds:
                print('Skipping conformer with different number of bonds')
                continue
            if not np.all(bond_index == data['bond_index']):
                print('Skipping conformer with different bond index')
                continue
            if not np.all(bond_type == data['bond_type']):
                print('Skipping conformer with different bond type')
                continue
        pos_all_confs.append(data['pos'])
        i_conf_list.append(i_conf)

    return {
        'element': np.array(element),
        'bond_index': np.array(bond_index),
        'bond_type': np.array(bond_type),
        # 'bond_rotatable': np.array(data['bond_rotatable']),
        'pos_all_confs': np.array(pos_all_confs, dtype=np.float32),
        'num_atoms': num_atoms,
        'num_bonds': num_bonds,
        'i_conf_list': i_conf_list,
        'num_confs': len(i_conf_list),
    }


def parse_3d_mol(mol, smiles=None, not_pos=False):
    if smiles is not None: # check smiles
        if smiles != Chem.MolToSmiles(mol, isomericSmiles=False):
            return None
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    if not not_pos:
        conf = mol.GetConformer()
    ele_list = []
    pos_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        ele = atom.GetAtomicNum()
        if not not_pos:
            pos = conf.GetAtomPosition(i)
            pos_list.append(list(pos))
        ele_list.append(ele)
    
    row, col = [], []
    bond_type = []
    for bond in mol.GetBonds():
        b_type = int(bond.GetBondType())
        assert b_type in [1, 2, 3, 12], 'Bond can only be 1,2,3,12 bond'
        b_type = b_type if b_type != 12 else 4
        b_index = [
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx()
        ]
        bond_type += 2*[b_type]
        row += [b_index[0], b_index[1]]
        col += [b_index[1], b_index[0]]
    
    bond_type = np.array(bond_type, dtype=np.int64)
    bond_index = np.array([row, col],dtype=np.int64)

    perm = (bond_index[0] * num_atoms + bond_index[1]).argsort()
    bond_index = bond_index[:, perm]
    bond_type = bond_type[perm]
    
    # # is rotable bond 
    # rot_mat = find_rotatable_bond_mat(mol)
    # bond_rotatable = rot_mat[bond_index[0], bond_index[1]]

    data = {
        'element': np.array(ele_list, dtype=np.int64),
        'pos': np.array(pos_list, dtype=np.float32),
        'bond_index': np.array(bond_index, dtype=np.int64),
        'bond_type': np.array(bond_type, dtype=np.int64),
        # 'bond_rotatable': np.array(bond_rotatable, dtype=np.int64),
        'num_atoms': num_atoms,
        'num_bonds': num_bonds,
    }
    return data


_CANONICAL_SC_FRAG_SMILES = frozenset([
    'C',                # Ala
    'C(C)C',            # Val
    'CC(C)C',           # Leu
    'C(C)CC',           # Ile
    'Cc1ccccc1',        # Phe
    'Cc1ccc(O)cc1',     # Tyr
    'Cc1c[nH]c2ccccc12',# Trp
    'Cc1cnc2ccccc21',   # Trp alt
    'CO',               # Ser
    'C(O)C',            # Thr
    'CS',               # Cys
    'CCSC',             # Met
    'CC(=O)N',          # Asn
    'CCC(=O)N',         # Gln
    'CC(=O)O',          # Asp
    'CCC(=O)O',         # Glu
    'CCCCN',            # Lys
    'CCCNC(=N)N',       # Arg
    'Cc1c[nH]cn1',      # His
    'Cc1cnc[nH]1',      # His tautomer
])

_NCAA_HETERO_ATOMIC_NUMS = frozenset([5, 9, 15, 17, 34, 35, 53])


def _heavy_neighbors(atom):
    return [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1]


def _carbonyl_oxygens(mol, c_idx):
    atom = mol.GetAtomWithIdx(c_idx)
    if atom.GetAtomicNum() != 6:
        return []
    out = []
    for nb in _heavy_neighbors(atom):
        if nb.GetAtomicNum() == 8 and len(_heavy_neighbors(nb)) == 1:
            bond = mol.GetBondBetweenAtoms(c_idx, nb.GetIdx())
            if bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
                out.append(nb.GetIdx())
    return out


def _is_carbonyl_c(mol, idx):
    return len(_carbonyl_oxygens(mol, idx)) > 0


def _perm_parity(src, dst):
    """Return +1 for even permutation, -1 for odd permutation."""
    idx_map = {v: i for i, v in enumerate(dst)}
    arr = [idx_map[v] for v in src]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return 1.0 if (inv % 2 == 0) else -1.0


def _ca_chirality_from_2d(mol, ca_idx, n_idx, c_idx, cb_idx):
    """Determine geometric L (+1) vs D (-1) sign from RDKit 2D ChiralTag without CIP R/S bias."""
    ca = mol.GetAtomWithIdx(ca_idx)
    tag = ca.GetChiralTag()
    if tag not in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
        return 0.0
    try:
        mol_h = Chem.AddHs(mol, onlyOnAtoms=[ca_idx])
        ca_h = mol_h.GetAtomWithIdx(ca_idx)
        nbr_order = [nb.GetIdx() for nb in ca_h.GetNeighbors()]
        if len(nbr_order) != 4:
            return 0.0
        h_nbs = [i for i in nbr_order if i not in (n_idx, c_idx, cb_idx)]
        if len(h_nbs) != 1:
            return 0.0
        h_idx = h_nbs[0]
        tag_h = ca_h.GetChiralTag()
        p_rdkit = 1.0 if tag_h == Chem.ChiralType.CHI_TETRAHEDRAL_CW else -1.0
        # Looking down H -> CA, L-AA has (N, C=O, CB) ordered clockwise (+1)
        p_perm = _perm_parity(nbr_order, [h_idx, n_idx, c_idx, cb_idx])
        return float(p_rdkit * p_perm)
    except Exception:
        return 0.0


def _find_ca_triple(mol, ca_idx, preferred_set):
    """If ca_idx is a peptide alpha-carbon, return (ca_idx, n_idx, c_idx) else None."""
    ca = mol.GetAtomWithIdx(ca_idx)
    if ca.GetAtomicNum() != 6 or ca.GetIsAromatic() or _is_carbonyl_c(mol, ca_idx):
        return None
    nbs = _heavy_neighbors(ca)
    n_cands = []
    c_cands = []
    for nb in nbs:
        nb_idx = nb.GetIdx()
        bond = mol.GetBondBetweenAtoms(ca_idx, nb_idx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if nb.GetAtomicNum() == 7 and not nb.GetIsAromatic():
            has_carb = any(_is_carbonyl_c(mol, nn.GetIdx()) for nn in _heavy_neighbors(nb) if nn.GetIdx() != ca_idx)
            n_cands.append((nb_idx in preferred_set, has_carb, -nb_idx, nb_idx))
        elif _is_carbonyl_c(mol, nb_idx):
            has_hetero = any(
                nn.GetAtomicNum() in (7, 8) and nn.GetIdx() not in _carbonyl_oxygens(mol, nb_idx)
                for nn in _heavy_neighbors(nb) if nn.GetIdx() != ca_idx
            )
            c_cands.append((nb_idx in preferred_set, has_hetero, -nb_idx, nb_idx))
    if not n_cands or not c_cands:
        return None
    n_cands.sort(reverse=True)
    c_cands.sort(reverse=True)
    n_idx = n_cands[0][-1]
    c_idx = c_cands[0][-1]
    if n_idx == c_idx:
        return None
    return (ca_idx, n_idx, c_idx)


def parse_cyclic_ncaa_mol(mol):
    """Parse a cyclic (or linear) peptide/ncAA RDKit molecule for Phase 2 training & sampling.

    Classifies macrocycle closure topology, partitions peptide backbone + minimal covalent
    cyclization bridge atoms (`peptide_is_backbone=True`) vs non-bridge sidechains and N-Me
    groups (`peptide_is_backbone=False`), marks `node_anchor_mask` on designable CA and amide N,
    and extracts 3D geometric supervision tensors:
      - `ca_anchor_triples`: `[M_ca, 3]` `[CA, N, C]`
      - `ca_chiral_quads`: `[M_chiral, 4]` `[CA, N, C, CB]` where
        `(x_N - x_CA) x (x_C - x_CA) . (x_CB - x_CA) > 0 <=> L-AA`
      - `amide_omega_quads`: `[M_omega, 4]` `[prev, C, N, next]`
      - `ring_closure_bonds`: `[M_close, 2]`
      - `ring_closure_target_len`: `[M_close]`
    """
    if mol is None:
        raise ValueError('mol cannot be None')
    ri = mol.GetRingInfo()
    if not getattr(ri, 'IsInitialized', lambda: ri.NumRings() > 0)():
        getattr(Chem, 'FastFindRings', Chem.GetSymmSSSR)(mol)

    n_atoms = mol.GetNumAtoms()
    has_conf = mol.GetNumConformers() > 0
    pos = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float32) if has_conf else np.zeros((n_atoms, 3), dtype=np.float32)
    is_3d = bool(has_conf and np.std(pos) > 1e-4)

    ring_info = mol.GetRingInfo()
    atom_rings = [list(r) for r in ring_info.AtomRings()]
    macro_rings = [r for r in atom_rings if len(r) >= 12]
    if not macro_rings:
        for r in atom_rings:
            if len(r) >= 9:
                r_set = set(r)
                amide_cnt = 0
                for b in mol.GetBonds():
                    u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                    if u in r_set and v in r_set:
                        if (_is_carbonyl_c(mol, u) and mol.GetAtomWithIdx(v).GetAtomicNum() == 7) or (
                            _is_carbonyl_c(mol, v) and mol.GetAtomWithIdx(u).GetAtomicNum() == 7
                        ):
                            amide_cnt += 1
                if amide_cnt >= 2:
                    macro_rings.append(r)

    has_macrocycle = len(macro_rings) > 0
    macro_union = set().union(*macro_rings) if has_macrocycle else set()
    primary_ring = list(max(macro_rings, key=len)) if has_macrocycle else []
    macrocycle_size = len(primary_ring)

    # 1. Identify peptide CA triples [CA, N, C]
    ca_triple_map = {}
    if has_macrocycle:
        for idx in macro_union:
            tr = _find_ca_triple(mol, idx, macro_union)
            if tr is not None:
                ca_triple_map[idx] = tr
        # Also extend along any exocyclic peptide backbone tails directly attached to macrocycle CAs
        queue = list(ca_triple_map.keys())
        while queue:
            cur_ca = queue.pop(0)
            _, cur_n, cur_c = ca_triple_map[cur_ca]
            # Look backward across cur_n -> prev_c -> prev_ca
            for nb_c in _heavy_neighbors(mol.GetAtomWithIdx(cur_n)):
                if nb_c.GetIdx() != cur_ca and _is_carbonyl_c(mol, nb_c.GetIdx()):
                    for nb_ca in _heavy_neighbors(nb_c):
                        cand = nb_ca.GetIdx()
                        if cand not in ca_triple_map and cand not in _carbonyl_oxygens(mol, nb_c.GetIdx()):
                            tr = _find_ca_triple(mol, cand, macro_union)
                            if tr is not None and tr[2] == nb_c.GetIdx():
                                ca_triple_map[cand] = tr
                                queue.append(cand)
            # Look forward across cur_c -> next_n -> next_ca
            for nb_n in _heavy_neighbors(mol.GetAtomWithIdx(cur_c)):
                if nb_n.GetIdx() != cur_ca and nb_n.GetAtomicNum() == 7:
                    for nb_ca in _heavy_neighbors(nb_n):
                        cand = nb_ca.GetIdx()
                        if cand not in ca_triple_map:
                            tr = _find_ca_triple(mol, cand, macro_union)
                            if tr is not None and tr[1] == nb_n.GetIdx():
                                ca_triple_map[cand] = tr
                                queue.append(cand)
    else:
        for idx in range(n_atoms):
            tr = _find_ca_triple(mol, idx, set(range(n_atoms)))
            if tr is not None:
                # Verify this CA participates in at least one amide linkage or multi-residue chain
                _, n_i, c_i = tr
                n_has_c = any(_is_carbonyl_c(mol, nn.GetIdx()) for nn in _heavy_neighbors(mol.GetAtomWithIdx(n_i)) if nn.GetIdx() != idx)
                c_has_n = any(nn.GetAtomicNum() == 7 for nn in _heavy_neighbors(mol.GetAtomWithIdx(c_i)) if nn.GetIdx() != idx)
                if n_has_c or c_has_n:
                    ca_triple_map[idx] = tr

    # Order CA triples cleanly in N->C backbone order
    # Build directed amide links CA_a -> CA_b where C_a bonds directly to N_b
    ca_next = {}
    ca_in_deg = {ca: 0 for ca in ca_triple_map}
    n_to_ca = {tr[1]: ca for ca, tr in ca_triple_map.items()}
    for ca_a, (_, _, c_a) in ca_triple_map.items():
        for nb in _heavy_neighbors(mol.GetAtomWithIdx(c_a)):
            nb_idx = nb.GetIdx()
            if nb_idx in n_to_ca and n_to_ca[nb_idx] != ca_a:
                ca_b = n_to_ca[nb_idx]
                ca_next.setdefault(ca_a, []).append(ca_b)
                ca_in_deg[ca_b] = ca_in_deg.get(ca_b, 0) + 1

    ordered_cas = []
    visited_cas = set()
    sources = sorted([ca for ca, deg in ca_in_deg.items() if deg == 0])
    if sources:
        # Linear / thioether / sidechain_lactam backbone chain: walk N->C from source(s)
        for src in sources:
            cur = src
            while cur is not None and cur not in visited_cas:
                ordered_cas.append(cur)
                visited_cas.add(cur)
                nxts = [c for c in sorted(ca_next.get(cur, [])) if c not in visited_cas]
                cur = nxts[0] if nxts else None
    elif has_macrocycle and primary_ring:
        # Pure head-to-tail cycle: walk primary_ring in N -> CA -> C direction from lowest-index N
        ring_len = len(primary_ring)
        for pos_i, a_idx in enumerate(primary_ring):
            if a_idx in ca_triple_map:
                _, n_i, c_i = ca_triple_map[a_idx]
                prev_a = primary_ring[(pos_i - 1) % ring_len]
                next_a = primary_ring[(pos_i + 1) % ring_len]
                if prev_a == c_i and next_a == n_i:
                    primary_ring = primary_ring[::-1]
                break
        start_candidates = []
        for pos_i, a_idx in enumerate(primary_ring):
            if a_idx in ca_triple_map:
                n_i = ca_triple_map[a_idx][1]
                start_candidates.append((n_i if n_i in primary_ring else a_idx, pos_i))
        if start_candidates:
            start_candidates.sort()
            rot = start_candidates[0][1]
            if primary_ring[(rot - 1) % ring_len] == ca_triple_map[primary_ring[rot]][1]:
                rot = (rot - 1) % ring_len
            primary_ring = primary_ring[rot:] + primary_ring[:rot]
        for a_idx in primary_ring:
            if a_idx in ca_triple_map and a_idx not in visited_cas:
                ordered_cas.append(a_idx)
                visited_cas.add(a_idx)
    for a_idx in sorted(ca_triple_map.keys()):
        if a_idx not in visited_cas:
            ordered_cas.append(a_idx)
            visited_cas.add(a_idx)

    ca_triples_list = [ca_triple_map[ca] for ca in ordered_cas]
    ca_anchor_triples = (
        np.asarray(ca_triples_list, dtype=np.int64) if ca_triples_list else np.zeros((0, 3), dtype=np.int64)
    )

    # 2. Classify closure_type
    def _in_small_ring(idx, max_size=6):
        return any(len(r) <= max_size and idx in r for r in atom_rings)

    backbone_core_set = set()
    for ca_i, n_i, c_i in ca_triples_list:
        backbone_core_set.update([ca_i, n_i, c_i])

    disulfide_bonds = []
    ester_bonds = []
    amide_bonds_in_macro = []
    thioether_atoms = []

    if has_macrocycle:
        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if u not in macro_union or v not in macro_union:
                continue
            au, av = mol.GetAtomWithIdx(u), mol.GetAtomWithIdx(v)
            zu, zv = au.GetAtomicNum(), av.GetAtomicNum()
            if zu == 16 and zv == 16 and not au.GetIsAromatic() and not av.GetIsAromatic():
                disulfide_bonds.append((u, v))
            elif _is_carbonyl_c(mol, u) and zv == 8 and not av.GetIsAromatic() and len(_heavy_neighbors(av)) >= 2:
                ester_bonds.append((u, v))
            elif _is_carbonyl_c(mol, v) and zu == 8 and not au.GetIsAromatic() and len(_heavy_neighbors(au)) >= 2:
                ester_bonds.append((v, u))
            elif _is_carbonyl_c(mol, u) and zv == 7 and not av.GetIsAromatic():
                amide_bonds_in_macro.append((u, v))
            elif _is_carbonyl_c(mol, v) and zu == 7 and not au.GetIsAromatic():
                amide_bonds_in_macro.append((v, u))

        for idx in sorted(macro_union):
            atom = mol.GetAtomWithIdx(idx)
            if (
                atom.GetAtomicNum() == 16
                and not atom.GetIsAromatic()
                and not _in_small_ring(idx, 6)
            ):
                nbs = _heavy_neighbors(atom)
                if len(nbs) == 2 and all(nb.GetAtomicNum() == 6 and nb.GetIdx() in macro_union for nb in nbs):
                    thioether_atoms.append(idx)

    bridge_atoms_in_macro = {idx for idx in macro_union if idx not in backbone_core_set}

    if not has_macrocycle:
        closure_type = 'other_macrocycle'
    elif disulfide_bonds or ester_bonds:
        closure_type = 'depsipeptide_or_disulfide'
    elif thioether_atoms:
        closure_type = 'thioether'
    elif amide_bonds_in_macro:
        if len(bridge_atoms_in_macro) == 0:
            closure_type = 'head_to_tail'
        elif any((c_i in bridge_atoms_in_macro) or (n_i in bridge_atoms_in_macro) for c_i, n_i in amide_bonds_in_macro):
            closure_type = 'sidechain_lactam'
        else:
            closure_type = 'other_macrocycle'
    else:
        closure_type = 'other_macrocycle'

    # 3. Mark peptide_is_backbone (node_bb): {N, CA, C, O} + minimal covalent bridge atoms + terminal caps
    is_bb = np.zeros(n_atoms, dtype=bool)
    for ca_i, n_i, c_i in ca_triples_list:
        is_bb[ca_i] = True
        is_bb[n_i] = True
        is_bb[c_i] = True
        for o_i in _carbonyl_oxygens(mol, c_i):
            is_bb[o_i] = True
        # Preserve terminal C-cap (-NH2 / -OH) directly on backbone carbonyl C (e.g. Cys8-CONH2)
        for nb in _heavy_neighbors(mol.GetAtomWithIdx(c_i)):
            nb_idx = nb.GetIdx()
            if nb_idx != ca_i and nb_idx not in _carbonyl_oxygens(mol, c_i):
                if len(_heavy_neighbors(nb)) == 1 and nb.GetAtomicNum() in (7, 8):
                    is_bb[nb_idx] = True
        # Preserve minimal N-terminal acetyl cap (CH3-C(=O)-NH-) if present
        for nb in _heavy_neighbors(mol.GetAtomWithIdx(n_i)):
            nb_idx = nb.GetIdx()
            if nb_idx != ca_i and nb_idx not in macro_union and _is_carbonyl_c(mol, nb_idx):
                c_nbs = [x for x in _heavy_neighbors(nb) if x.GetIdx() != n_i and x.GetIdx() not in _carbonyl_oxygens(mol, nb_idx)]
                if len(c_nbs) == 1 and c_nbs[0].GetAtomicNum() == 6 and len(_heavy_neighbors(c_nbs[0])) == 1:
                    is_bb[nb_idx] = True
                    is_bb[c_nbs[0].GetIdx()] = True
                    for o_i in _carbonyl_oxygens(mol, nb_idx):
                        is_bb[o_i] = True

    if has_macrocycle:
        for idx in macro_union:
            is_bb[idx] = True
            for o_i in _carbonyl_oxygens(mol, idx):
                is_bb[o_i] = True
            # If bridge passes through a small aromatic ring (e.g. xylene staple), keep that ring intact in node_bb
            if idx in bridge_atoms_in_macro and mol.GetAtomWithIdx(idx).GetIsAromatic():
                for r in atom_rings:
                    if len(r) <= 6 and idx in r and all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r):
                        for a in r:
                            is_bb[a] = True

    # 4. Mark node_anchor_mask on designable CA and backbone amide N
    node_anchor_mask = np.zeros(n_atoms, dtype=bool)
    for ca_i, n_i, c_i in ca_triples_list:
        ca_bb_nbs = [nb.GetIdx() for nb in _heavy_neighbors(mol.GetAtomWithIdx(ca_i)) if is_bb[nb.GetIdx()]]
        # Designable CA only has {N, C} in node_bb (bridge-junction CAs also have bridge CB in node_bb)
        if len(ca_bb_nbs) <= 2 and set(ca_bb_nbs).issubset({n_i, c_i}):
            node_anchor_mask[ca_i] = True
        # Designable backbone amide N must be bonded to a carbonyl C (not primary terminal NH2)
        n_has_carb = any(_is_carbonyl_c(mol, nb.GetIdx()) and is_bb[nb.GetIdx()] for nb in _heavy_neighbors(mol.GetAtomWithIdx(n_i)))
        if n_has_carb:
            node_anchor_mask[n_i] = True

    # 5. Extract ca_chiral_quads [CA, N, C, CB] & ca_chiral_signs
    chiral_quads = []
    ca_chiral_signs = []
    for ca_i, n_i, c_i in ca_triples_list:
        sc_nbs = [nb.GetIdx() for nb in _heavy_neighbors(mol.GetAtomWithIdx(ca_i)) if nb.GetIdx() not in (n_i, c_i)]
        if len(sc_nbs) == 1:
            cb_i = sc_nbs[0]
            chiral_quads.append([ca_i, n_i, c_i, cb_i])
            if is_3d:
                v_n = pos[n_i] - pos[ca_i]
                v_c = pos[c_i] - pos[ca_i]
                v_cb = pos[cb_i] - pos[ca_i]
                vol = float(np.dot(np.cross(v_n, v_c), v_cb))
                if abs(vol) > 1e-4:
                    sign = 1.0 if vol > 0.0 else -1.0
                else:
                    sign = _ca_chirality_from_2d(mol, ca_i, n_i, c_i, cb_i)
            else:
                sign = _ca_chirality_from_2d(mol, ca_i, n_i, c_i, cb_i)
            ca_chiral_signs.append(sign)
        else:
            # Gly (0 heavy SC neighbors) or symmetric quaternary (e.g. Aib)
            ca_chiral_signs.append(0.0)

    ca_chiral_quads = (
        np.asarray(chiral_quads, dtype=np.int64) if chiral_quads else np.zeros((0, 4), dtype=np.int64)
    )
    ca_chiral_signs_arr = np.asarray(ca_chiral_signs, dtype=np.float32)

    # 6. Extract amide_omega_quads [prev, C, N, next] along backbone/macrocycle amides
    omega_quads = []
    seen_cn = set()
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if not (is_bb[u] and is_bb[v]):
            continue
        c_i, n_i = None, None
        if _is_carbonyl_c(mol, u) and mol.GetAtomWithIdx(v).GetAtomicNum() == 7:
            c_i, n_i = u, v
        elif _is_carbonyl_c(mol, v) and mol.GetAtomWithIdx(u).GetAtomicNum() == 7:
            c_i, n_i = v, u
        if c_i is None or (c_i, n_i) in seen_cn:
            continue
        seen_cn.add((c_i, n_i))
        o_set = set(_carbonyl_oxygens(mol, c_i))
        prev_cands = [
            nb.GetIdx() for nb in _heavy_neighbors(mol.GetAtomWithIdx(c_i))
            if nb.GetIdx() != n_i and nb.GetIdx() not in o_set and is_bb[nb.GetIdx()]
        ]
        next_cands = [
            nb.GetIdx() for nb in _heavy_neighbors(mol.GetAtomWithIdx(n_i))
            if nb.GetIdx() != c_i and is_bb[nb.GetIdx()]
        ]
        if prev_cands and next_cands:
            omega_quads.append([prev_cands[0], c_i, n_i, next_cands[0]])
    amide_omega_quads = (
        np.asarray(omega_quads, dtype=np.int64) if omega_quads else np.zeros((0, 4), dtype=np.int64)
    )

    # 7. Extract ring_closure_bonds [M, 2] & ring_closure_target_len [M]
    closure_bonds = []
    closure_lens = []
    if has_macrocycle:
        if closure_type == 'depsipeptide_or_disulfide':
            for u, v in disulfide_bonds:
                closure_bonds.append([u, v])
                closure_lens.append(2.04)
            for c_i, o_i in ester_bonds:
                closure_bonds.append([c_i, o_i])
                closure_lens.append(1.345)
        elif closure_type == 'thioether':
            for s_i in thioether_atoms:
                for nb in _heavy_neighbors(mol.GetAtomWithIdx(s_i)):
                    if nb.GetIdx() in macro_union:
                        closure_bonds.append([s_i, nb.GetIdx()])
                        closure_lens.append(1.815)
        elif closure_type == 'sidechain_lactam':
            for c_i, n_i in amide_bonds_in_macro:
                if (c_i in bridge_atoms_in_macro) or (n_i in bridge_atoms_in_macro):
                    closure_bonds.append([c_i, n_i])
                    closure_lens.append(1.335)
        elif closure_type == 'head_to_tail':
            if len(ca_triples_list) >= 2:
                c_last = ca_triples_list[-1][2]
                n_first = ca_triples_list[0][1]
                if mol.GetBondBetweenAtoms(c_last, n_first) is not None:
                    closure_bonds.append([c_last, n_first])
                    closure_lens.append(1.335)
            if not closure_bonds and amide_bonds_in_macro:
                c_i, n_i = amide_bonds_in_macro[-1]
                closure_bonds.append([c_i, n_i])
                closure_lens.append(1.335)
        else:
            if amide_bonds_in_macro:
                c_i, n_i = amide_bonds_in_macro[-1]
                closure_bonds.append([c_i, n_i])
                closure_lens.append(1.335)
            elif len(primary_ring) >= 2:
                closure_bonds.append([primary_ring[-1], primary_ring[0]])
                closure_lens.append(1.53)

    ring_closure_bonds = (
        np.asarray(closure_bonds, dtype=np.int64) if closure_bonds else np.zeros((0, 2), dtype=np.int64)
    )
    ring_closure_target_len = (
        np.asarray(closure_lens, dtype=np.float32) if closure_lens else np.zeros((0,), dtype=np.float32)
    )

    # 8. Residue assignment & atom names (compatible with VariableScSize / PepdesignTransform)
    peptide_atom_name = ['BRG' if is_bb[i] else 'SC' for i in range(n_atoms)]
    peptide_res_index = np.full(n_atoms, -1, dtype=np.int64)
    bfs_q = []
    for r_idx, (ca_i, n_i, c_i) in enumerate(ca_triples_list):
        peptide_atom_name[n_i] = 'N'
        peptide_atom_name[ca_i] = 'CA'
        peptide_atom_name[c_i] = 'C'
        for a_i in (n_i, ca_i, c_i):
            if peptide_res_index[a_i] < 0:
                peptide_res_index[a_i] = r_idx
                bfs_q.append(a_i)
        for o_i in _carbonyl_oxygens(mol, c_i):
            peptide_atom_name[o_i] = 'O'
            if peptide_res_index[o_i] < 0:
                peptide_res_index[o_i] = r_idx
                bfs_q.append(o_i)

    if len(ca_triples_list) > 0:
        while bfs_q:
            cur = bfs_q.pop(0)
            cur_r = peptide_res_index[cur]
            for nb in _heavy_neighbors(mol.GetAtomWithIdx(cur)):
                nb_idx = nb.GetIdx()
                if peptide_res_index[nb_idx] < 0:
                    peptide_res_index[nb_idx] = cur_r
                    bfs_q.append(nb_idx)
        peptide_res_index = np.where(peptide_res_index < 0, 0, peptide_res_index)
        peptide_pep_len = len(ca_triples_list)
    else:
        peptide_res_index = np.zeros(n_atoms, dtype=np.int64)
        peptide_pep_len = 1

    # 9. Detect D-AA, N-Me/N-alkyl, halogen/hetero ncAA, and canonical all-L status
    has_d_aa = bool(np.any(ca_chiral_signs_arr < 0))
    has_halogen_or_hetero = any(
        atom.GetAtomicNum() in _NCAA_HETERO_ATOMIC_NUMS for atom in mol.GetAtoms()
    )
    pro_5rings = [set(r) for r in atom_rings if len(r) == 5]
    has_nme = False
    for ca_i, n_i, c_i in ca_triples_list:
        n_nbs = _heavy_neighbors(mol.GetAtomWithIdx(n_i))
        if len(n_nbs) >= 3:
            extra_nbs = [nb.GetIdx() for nb in n_nbs if nb.GetIdx() != ca_i and not _is_carbonyl_c(mol, nb.GetIdx())]
            for ex_idx in extra_nbs:
                is_pro = any((n_i in r and ca_i in r and ex_idx in r) for r in pro_5rings)
                if not is_pro:
                    has_nme = True
                    break
        if has_nme:
            break

    is_canonical_all_l = False
    if (
        closure_type == 'head_to_tail'
        and has_macrocycle
        and len(ca_triples_list) > 0
        and not has_d_aa
        and not has_nme
        and not has_halogen_or_hetero
        and all(a.GetAtomicNum() in (1, 6, 7, 8, 16) for a in mol.GetAtoms())
    ):
        is_canonical_all_l = True
        for ca_i, n_i, c_i in ca_triples_list:
            sc_roots = [nb.GetIdx() for nb in _heavy_neighbors(mol.GetAtomWithIdx(ca_i)) if nb.GetIdx() not in (n_i, c_i)]
            if not sc_roots:
                continue  # Gly
            if len(sc_roots) > 1:
                is_canonical_all_l = False
                break
            cb_i = sc_roots[0]
            # Check Proline 5-ring
            if any((ca_i in r and n_i in r and cb_i in r) for r in pro_5rings):
                pro_ring = next(r for r in pro_5rings if (ca_i in r and n_i in r and cb_i in r))
                # Ensure no extra substitution on Proline ring atoms (besides C=O on CA)
                extra_on_pro = any(
                    any(nb.GetIdx() not in pro_ring and nb.GetIdx() != c_i and not _is_carbonyl_c(mol, nb.GetIdx())
                        for nb in _heavy_neighbors(mol.GetAtomWithIdx(p_a)))
                    for p_a in pro_ring
                )
                if extra_on_pro:
                    is_canonical_all_l = False
                    break
                continue
            # BFS sidechain atoms from cb_i stopping at ca_i
            sc_atoms = {cb_i}
            sc_q = [cb_i]
            while sc_q:
                curr = sc_q.pop(0)
                for nb in _heavy_neighbors(mol.GetAtomWithIdx(curr)):
                    nb_idx = nb.GetIdx()
                    if nb_idx != ca_i and nb_idx not in sc_atoms:
                        if is_bb[nb_idx]:
                            # Linked back to backbone (not Pro)
                            is_canonical_all_l = False
                            break
                        sc_atoms.add(nb_idx)
                        sc_q.append(nb_idx)
                if not is_canonical_all_l:
                    break
            if not is_canonical_all_l:
                break
            try:
                sc_smi = Chem.MolFragmentToSmiles(
                    mol, atomsToUse=sorted(sc_atoms), rootedAtAtom=cb_i,
                    canonical=True, isomericSmiles=False
                )
                if sc_smi not in _CANONICAL_SC_FRAG_SMILES:
                    is_canonical_all_l = False
                    break
            except Exception:
                is_canonical_all_l = False
                break

    peptide_seq = ('G' * peptide_pep_len) if is_canonical_all_l else ('X' * peptide_pep_len)

    return {
        'closure_type': closure_type,
        'has_macrocycle': has_macrocycle,
        'macrocycle_size': int(macrocycle_size),
        'peptide_pos': pos,
        'peptide_atom_name': peptide_atom_name,
        'peptide_res_index': peptide_res_index,
        'peptide_is_backbone': is_bb,
        'peptide_pep_len': int(peptide_pep_len),
        'peptide_seq': peptide_seq,
        'node_anchor_mask': node_anchor_mask,
        'ca_anchor_triples': ca_anchor_triples,
        'ca_chiral_quads': ca_chiral_quads,
        'ca_chiral_signs': ca_chiral_signs_arr,
        'amide_omega_quads': amide_omega_quads,
        'ring_closure_bonds': ring_closure_bonds,
        'ring_closure_target_len': ring_closure_target_len,
        'has_d_aa': has_d_aa,
        'has_nme': has_nme,
        'has_halogen_or_hetero': has_halogen_or_hetero,
        'is_canonical_all_l': is_canonical_all_l,
    }