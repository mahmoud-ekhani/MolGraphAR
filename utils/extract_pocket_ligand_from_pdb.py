import argparse
import logging
from rdkit import Chem
from utils.protein_ligand import PDBProtein, PDBLigand

def parse_arguments():
    """
    Parses command line arguments.
    
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Process PDB files for docking.')
    parser.add_argument('--pdb_path', type=str, default='data/cresset_spark_capn/Spark_reference_pocket20.pdb', help='Path to the PDB file.')
    parser.add_argument('--radius', type=float, default=20.0, help='Cut-off threshold from the ligand center.')
    return parser.parse_args()

def read_file(file_path):
    """
    Reads the content of a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Content of the file.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except IOError as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None

def write_sdf(file_path, rdmol):
    """
    Writes an RDKit molecule object to an SDF file.

    Args:
        file_path (str): Path to save the SDF file.
        rdmol (rdkit.Chem.rdchem.Mol): Molecule object.

    Returns:
        None
    """
    try:
        with Chem.SDWriter(file_path) as sdf_writer:
            sdf_writer.write(rdmol)
        logging.info(f'SDF file created at {file_path}')
    except Exception as e:
        logging.error(f"Error writing SDF file {file_path}: {e}")

def main():
    """
    Main function to process PDB files for docking.
    """
    args = parse_arguments()

    pdb_block = read_file(args.pdb_path)
    if pdb_block is None:
        return

    protein = PDBProtein(pdb_block)
    ligand = PDBLigand(pdb_block)

    protein_pocket = protein.query_residues_ligand(ligand.to_dict_atom(), args.radius)
    pocket_pdb_block = protein.residues_to_pdb_block(protein_pocket)

    pocket_file_path = f'{args.pdb_path.split(".")[0]}_pocket{round(args.radius)}.pdb'
    try:
        with open(pocket_file_path, 'w') as file:
            file.write(pocket_pdb_block)
        logging.info(f'Pocket PDB file saved at {pocket_file_path}')
    except IOError as e:
        logging.error(f"Error writing pocket PDB file {pocket_file_path}: {e}")

    ligand_pdb_block = ligand.atoms_to_pdb_block()
    rdmol = Chem.MolFromPDBBlock(ligand_pdb_block, removeHs=False)
    if rdmol:
        logging.info(f"SMILES String of the extracted ligand: {Chem.MolToSmiles(rdmol)}")
        ligand_file_path = f'{args.pdb_path.split(".")[0]}_ligand.sdf'
        write_sdf(ligand_file_path, rdmol)
    else:
        logging.error("Failed to convert PDB block to RDKit molecule.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
