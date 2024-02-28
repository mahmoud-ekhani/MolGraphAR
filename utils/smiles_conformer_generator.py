import argparse
import os
import subprocess
import tempfile
import contextlib
import string
import random
from openbabel import pybel
from meeko import MoleculePreparation, obutils
from vina import Vina
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import AutoDockTools

def suppress_stdout(func):
    def wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*args, **kwargs)
    return wrapper

def get_random_id(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length)) 

class PrepLig:
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf': 
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'Unsupported molecule format: {mol_format}')
        
    def add_hydrogens(self, polar_only=False, correct_for_ph=True, ph=7):
        self.ob_mol.OBMol.AddHydrogens(polar_only, correct_for_ph, ph)
        obutils.writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')

    def generate_conformation(self):
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        
        # Energy minimization
        force_field = AllChem.UFFGetMoleculeForceField(rdkit_mol)
        if force_field:
            force_field.Initialize()
            force_field.Minimize()
        
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

    @suppress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt:
            preparator.write_pdbqt_file(lig_pdbqt)
        else:
            return preparator.write_pdbqt_string()
        

class PrepProt:
    def __init__(self, pdb_file): 
        self.prot = pdb_file
    
    def delete_water(self, dry_pdb_file):
        with open(self.prot) as file:
            lines = [line for line in file if line.startswith('ATOM') or line.startswith('HETATM')]
            dry_lines = [line for line in lines if 'HOH' not in line]

        with open(dry_pdb_file, 'w') as file:
            file.writelines(dry_lines)
        self.prot = dry_pdb_file
        
    def add_hydrogens(self, prot_pqr):
        self.prot_pqr = prot_pqr
        subprocess.run(['pdb2pqr30', '--ff=AMBER', self.prot, self.prot_pqr], 
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        subprocess.run(['python3', prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt], 
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


class VinaDock(object): 
    def __init__(self, lig_pdbqt, prot_pdbqt): 
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
    
    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size
    
    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket. 
        buffer: buffer size to add 

        if ref is not None: 
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
        else: 
            use the entire protein to define pocket 
        '''
        if ref is None: 
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=False, **kwargs):  # seed=0 mean random seed
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == 'score_only': 
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
        else:
            raise ValueError
        
        if not save_pose: 
            return score
        else: 
            if mode == 'score_only': 
                pose = None 
            elif mode == 'minimize': 
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, 'w') as f: 
                    v.write_pose(tmp.name, overwrite=True)             
                with open(tmp.name, 'r') as f: 
                    pose = f.read()
   
            elif mode == 'dock': 
                pose = v.poses(n_poses=1)
            else:
                raise ValueError
            return score, pose


class VinaDockingTask:

    def __init__(self, 
                 smiles,
                 protein_path, 
                 ref_ligand_path, 
                 tmp_dir='./tmp', 
                 center=None, 
                 size_factor=None, 
                 buffer=5.0):

        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.smiles = smiles

        ref_ligand_rdmol = next(iter(Chem.SDMolSupplier(ref_ligand_path)))
        # self.recon_ligand_mol = ligand_rdmol
        ref_ligand_rdmol = Chem.AddHs(ref_ligand_rdmol, addCoords=True)
        self.ref_ligand_rdmol = ref_ligand_rdmol
        self.protein_path = protein_path

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'
        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_id + '.pdb')
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        # sdf_writer = Chem.SDWriter(self.ligand_path)
        # sdf_writer.write(ligand_rdmol)
        # sdf_writer.close()

        pos = ref_ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

    def run(self, mode='dock', exhaustiveness=8, **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_pqr = self.receptor_path[:-4] + '.pqr'
        protein_pdbqt = self.receptor_path[:-4] + '.pdbqt'

        lig = PrepLig(self.smiles, 'smi')
        lig.get_pdbqt(ligand_pdbqt)

        prot = PrepProt(self.protein_path)
        if not os.path.exists(protein_pqr):
            prot.addH(protein_pqr)
        if not os.path.exists(protein_pdbqt):
            prot.get_pdbqt(protein_pdbqt)

        dock = VinaDock(ligand_pdbqt, protein_pdbqt)
        dock.pocket_center, dock.box_size = self.center, [self.size_x, self.size_y, self.size_z]
        score, pose = dock.dock(score_func='vina', mode=mode, exhaustiveness=exhaustiveness, save_pose=True, **kwargs)
        return {'affinity': score, 'pose': pose}
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Molecular docking using SMILES and PDB files.')
    parser.add_argument('--smiles', type=str, required=True, help='SMILES string of the ligand.')
    parser.add_argument('--pdb_path', type=str, required=True, help='Path to the protein PDB file.')
    parser.add_argument('--sdf_path', type=str, required=True, help='Path to the reference ligand SDF file.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    docking_task = VinaDockingTask(smiles=args.smiles,
                                   protein_path=args.pdb_path,
                                   ref_ligand_path=args.sdf_path)
    result = docking_task.run()
    print(f'Docking score: {result["score"]}')
    if result["pose"]:
        print('Docked pose:', result["pose"])
    