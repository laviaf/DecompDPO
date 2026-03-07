from rdkit import Chem
from rdkit.Chem.BRICS import FindBRICSBonds
from utils.fragmentation.utils import get_rings, get_other_atom_idx, find_parts_bonds
from rdkit.Chem.rdchem import BondType
import networkx as nx
import copy


def get_fragmentizer(method):
    if method == 'rot_bond':
        sg = RotBondFragmentizer()
    elif method == 'brics_cut_single_bond':
        sg = BRICS_RING_R_Fragmentizer()
    else:
        raise NotImplementedError
    print(f'{sg.type} fragmentizer is used!')
    return sg


class BRICS_Fragmentizer():
    def __inti__(self):
        self.type = 'BRICS_Fragmentizers'
    
    def get_bonds(self, mol):
        bonds = [bond[0] for bond in list(FindBRICSBonds(mol))]
        return bonds
    
    def fragmentize(self, mol, dummyStart=1):
        # get bonds need to be break
        bonds = [bond[0] for bond in list(FindBRICSBonds(mol))]
        
        # whether the molecule can really be break
        if len(bonds) != 0:
            bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]

            # break the bonds & set the dummy labels for the bonds
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1

        return break_mol, dummyEnd


class RING_R_Fragmentizer():
    def __init__(self):
        self.type = 'RING_R_Fragmentizer'

    def bonds_filter(self, mol, bonds):
        filted_bonds = []
        for bond in bonds:
            bond_type = mol.GetBondBetweenAtoms(bond[0], bond[1]).GetBondType()
            if not bond_type is BondType.SINGLE:
                continue
            f_atom = mol.GetAtomWithIdx(bond[0])
            s_atom = mol.GetAtomWithIdx(bond[1])
            if f_atom.GetSymbol() == '*' or s_atom.GetSymbol() == '*':
                continue
            if mol.GetBondBetweenAtoms(bond[0], bond[1]).IsInRing():
                continue
            filted_bonds.append(bond)
        return filted_bonds

    def get_bonds(self, mol):
        bonds = []
        rings = get_rings(mol)
        if len(rings) > 0:
            for ring in rings:
                rest_atom_idx = get_other_atom_idx(mol, ring)
                bonds += find_parts_bonds(mol, [rest_atom_idx, ring])
            bonds = self.bonds_filter(mol, bonds)
        return bonds

    def fragmentize(self, mol, dummyStart=1):
        rings = get_rings(mol)
        if len(rings) > 0:
            bonds = []
            for ring in rings:
                rest_atom_idx = get_other_atom_idx(mol, ring)
                bonds += find_parts_bonds(mol, [rest_atom_idx, ring])
            bonds = self.bonds_filter(mol, bonds)
            if len(bonds) > 0:
                bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
                bond_ids = list(set(bond_ids))
                dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
                break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
                dummyEnd = dummyStart + len(dummyLabels) - 1
            else:
                break_mol = mol
                dummyEnd = dummyStart - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1
        return break_mol, dummyEnd


class BRICS_RING_R_Fragmentizer():
    def __init__(self):
        self.type = 'BRICS_RING_R_Fragmentizer'
        self.brics_fragmenizer = BRICS_Fragmentizer()
        self.ring_r_fragmenizer = RING_R_Fragmentizer()

    def fragmentize(self, mol, dummyStart=1):
        brics_bonds = self.brics_fragmenizer.get_bonds(mol)
        ring_r_bonds = self.ring_r_fragmenizer.get_bonds(mol)
        bonds = brics_bonds + ring_r_bonds

        if len(bonds) != 0:
            bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
            bond_ids = list(set(bond_ids))
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            bond_ids = []
            dummyEnd = dummyStart - 1

        return break_mol, bond_ids, dummyEnd


class RotBondFragmentizer():
    def __init__(self, only_single_bond=True):
        self.type = 'RotBondFragmentizer'
        self.only_single_bond = only_single_bond

    # code adapt from Torsion Diffusion
    def get_bonds(self, mol):
        bonds = []
        G = nx.Graph()
        for i, atom in enumerate(mol.GetAtoms()):
            G.add_node(i)
        # nodes = set(G.nodes())
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            G.add_edge(start, end)
        for e in G.edges():
            G2 = copy.deepcopy(G)
            G2.remove_edge(*e)
            if nx.is_connected(G2): continue
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) < 2: continue
            # n0 = list(G2.neighbors(e[0]))
            # n1 = list(G2.neighbors(e[1]))
            if self.only_single_bond:
                bond_type = mol.GetBondBetweenAtoms(e[0], e[1]).GetBondType()
                if bond_type != BondType.SINGLE:
                    continue
            bonds.append((e[0], e[1]))
        return bonds

    def fragmentize(self, mol, dummyStart=1, bond_list=None):
        if bond_list is None:
            # get bonds need to be break
            bonds = self.get_bonds(mol)
        else:
            bonds = bond_list
        # whether the molecule can really be break
        if len(bonds) != 0:
            bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
            bond_ids = list(set(bond_ids))
            # break the bonds & set the dummy labels for the bonds
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            bond_ids = []
            dummyEnd = dummyStart - 1

        return break_mol, bond_ids, dummyEnd



if __name__ == "__main__":
    # test_smiles = 'COc1cccc(O[C@@H]2CCC[N@@H+](Cc3cnn(C)c3)C2)c1'
    # test_smiles = 'c1ccccc1'
    test_smiles = 'CC'
    test_mol = Chem.MolFromSmiles(test_smiles)
    
    fragmentizer = BRICS_Fragmentizer()
    frag, _ = fragmentizer.fragmentize(test_mol)
