import argparse
import os
import subprocess
import tempfile
import contextlib
import string
import random
import shutil
import pandas as pd
from tqdm.auto import tqdm
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
        
        # Attempt to generate 3D conformation with RDKit
        if AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3()) == 0:
            # RDKit conformation generation succeeded
            AllChem.UFFOptimizeMolecule(rdkit_mol)
            self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        else:
            # RDKit conformation generation failed, use Open Babel
            # This requires a temporary file
            tmp_filename = 'temp_molecule.sdf'
            self.ob_mol.write('sdf', tmp_filename)
            subprocess.run(['obabel', tmp_filename, '-O', tmp_filename, '--gen3D'])
            self.ob_mol = next(pybel.readfile('sdf', tmp_filename))
            os.remove(tmp_filename)
        
        obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')
        os.remove('tmp_h.sdf')

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
    def __init__(self, lig_pdbqt, prot_pdbqt, lig_sdf): 
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
        self.lig_sdf = lig_sdf
    
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

    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, n_poses=1, save_pose=False, **kwargs):
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)

        if mode == 'score_only': 
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            score = v.energies(n_poses=n_poses)[0][0]
        else:
            raise ValueError

        if save_pose: 
            if mode in ['minimize', 'dock']:
                # Save docked poses in PDBQT format
                v.write_poses(self.lig_pdbqt, n_poses=n_poses, overwrite=True)
                
                # Convert PDBQT to SDF
                sdf_filename = self.lig_sdf
                subprocess.run(['obabel', '-ipdbqt', self.lig_pdbqt, '-osdf', '-O' + sdf_filename],
                            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

            if mode == 'score_only':
                pose = None
            else:
                # Read back the converted SDF file
                with open(sdf_filename, 'r') as f:
                    pose = f.read()

            return score, pose
        else:
            return score


class VinaDockingTask:
    def __init__(self, 
                 smiles,
                 protein_path, 
                 ref_ligand_path, 
                 gen_ligand_name=None,
                 tmp_dir='./tmp', 
                 center=None, 
                 size_factor=None, 
                 buffer=5.0):

        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.smiles = smiles

        ref_ligand_rdmol = next(iter(Chem.SDMolSupplier(ref_ligand_path)))
        ref_ligand_rdmol = Chem.AddHs(ref_ligand_rdmol, addCoords=True)
        self.ref_ligand_rdmol = ref_ligand_rdmol
        self.protein_path = protein_path

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'
        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_id + '.pdb')
        if gen_ligand_name is None:
            self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')
        else:
            self.ligand_path = os.path.join(self.tmp_dir, gen_ligand_name + '.sdf')


        pos = ref_ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 30, 30, 30
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

    def run(self, mode='dock', exhaustiveness=8, n_poses=1, **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_dry = self.receptor_path[:-4] + '.pdb'
        protein_pqr = self.receptor_path[:-4] + '.pqr'
        protein_pdbqt = self.receptor_path[:-4] + '.pdbqt'
        ligand_sdf = self.ligand_path

        lig = PrepLig(self.smiles, 'smi')
        lig.add_hydrogens()
        lig.generate_conformation()
        lig.get_pdbqt(ligand_pdbqt)

        prot = PrepProt(self.protein_path)
        prot.delete_water(protein_dry)
        prot.add_hydrogens(protein_pqr)
        prot.get_pdbqt(protein_pdbqt)

        dock = VinaDock(ligand_pdbqt, protein_pdbqt, ligand_sdf)
        dock.pocket_center, dock.box_size = self.center, [self.size_x, self.size_y, self.size_z]
        score, pose = dock.dock(score_func='vina', mode=mode, exhaustiveness=exhaustiveness, n_poses=n_poses, save_pose=True, **kwargs)
        os.remove('conf_h.sdf')
        os.remove(ligand_pdbqt)
        os.remove(protein_dry)
        os.remove(protein_pqr)
        os.remove(protein_pdbqt)
        return {'affinity': score, 'pose': pose}
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Molecular docking using SMILES and PDB files.')
    parser.add_argument('--smiles', type=str, default='NC(=O)C(=O)C(Cc1ccccc1)NC(=O)C', help='SMILES string of the ligand.')
    parser.add_argument('--mol_name', type=str, default='1910-1552-modified', help='Name of the molecule associated with the smiles.')
    parser.add_argument('--pdb_path', type=str, default='example/7x79_pocket10.pdb', help='Path to the protein PDB file.')
    parser.add_argument('--sdf_path', type=str, default='example/7x79_ligand.sdf', help='Path to the reference ligand SDF file.')
    parser.add_argument('--csv_path', type=str, default=None, help='Path to the csv file for batch processing.')
    parser.add_argument('--n_poses', type=int, default=1, help='number of generated binding poses.')
    parser.add_argument('--exhaustiveness', type=int, default=20, help='controls how extensively the docking algorithm explores the possible orientations and conformations')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if args.csv_path is None:
        docking_task = VinaDockingTask(smiles=args.smiles,
                                    protein_path=args.pdb_path,
                                    ref_ligand_path=args.sdf_path,
                                    )
        result = docking_task.run(exhaustiveness=args.exhaustiveness, n_poses=args.n_poses)
        print(f'Docking score: {result["affinity"]}')
        for filename in os.listdir('tmp'):
                if filename.endswith('.sdf'):
                    shutil.move(os.path.join('tmp', filename), 
                                os.path.join(os.path.dirname(args.pdb_path), args.mol_name + '.sdf')
                                )
        shutil.rmtree('tmp')

    else:
        df = pd.read_csv(args.csv_path)
        for index, row in tqdm(df.iterrows(), desc='Docking Molecules', total=df.shape[0]):
            molecule_name = row['Molecule Name']
            smiles = row['SMILES']
            docking_task = VinaDockingTask(smiles=smiles,
                                        protein_path=args.pdb_path,
                                        ref_ligand_path=args.sdf_path,
                                        gen_ligand_name=molecule_name
                                        )
                                        
            result = docking_task.run(exhaustiveness=args.exhaustiveness, n_poses=args.n_poses)
            print(f'Docking score: {result["affinity"]}')

            csv_file_name = os.path.basename(args.csv_path)
            new_folder_name = csv_file_name.replace('.csv', '_conformers')
            new_folder_path = os.path.join(os.path.dirname(args.csv_path), new_folder_name)

            os.makedirs(new_folder_path, exist_ok=True)

            for filename in os.listdir('tmp'):
                if filename.endswith('.sdf'):
                    shutil.move(os.path.join('tmp', filename), new_folder_path)
            shutil.rmtree('tmp')

    