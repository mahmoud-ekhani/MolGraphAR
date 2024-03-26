# MolGraphAR: Autoregressive Molecular Fragment Growing in 3D Using Pocket2Mol

<!-- [Pocket2Mol](https://arxiv.org/abs/2205.07249) used equivariant graph neural networks to improve efficiency and molecule quality of [previous structure-based drug design model](https://arxiv.org/abs/2203.10446).

<img src="./assets/model.jpg" alt="model"  width="70%"/> -->


## Installation
``` bash
conda create -n molgraphar python=3.8
conda activate molgraphar

# Install PyTorch and PyTorch Geometric
conda install pytorch==1.10.1 -c pytorch
conda install pyg -c pyg
conda install pytorch-scatter -c pyg

# Install RdKit, BioPython, OpenBable, and other tools
conda install rdkit -c conda-forge 
conda install biopython -c conda-forge 
conda install openbabel -c conda-forge
conda install pyyaml easydict python-lmdb -c conda-forge

# Install docking tools
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2
pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

## Extract Pocket and Ligand from PDB File

You can extract the protein pocket and the corresponding ligand from a PDB file using the script `utils/extract_pocket_ligand_from_pdb.py`. 

### Features
- **Protein Pocket Extraction**: Specify a radius to extract residues within a certain distance from the ligand center.
- **Output Formats**: The extracted pocket and ligand are saved in `.pdb` and `.sdf` formats, respectively.

### Usage

To use the script, provide the path to the PDB file and the desired radius for pocket extraction. For example, to process the file `example/7x79.pdb` with a radius of 10.0 Å, use the following command:

```bash
python utils/extract_pocket_ligand_from_pdb.py --pdb_path example/7x79.pdb --radius 10.0
```

The outputs will be saved in the same directory as the pdb file, e.g., `example/7x79_pocket10.pdb` contains the pocket and `example/7x79_ligand.sdf` contains the cognate ligand.

## Generating a docked conformer from a given smiles

You can create a conformer for a given SMILES string using the `utils/smiles_conformer_generator.py` tool.

You need to specify the protein pocket and a reference molecule.

A general conformer is first generated using OpenBabel.

The reference molecule's center will be used to determine the bounding box for docking the conformer of the SMILES to the protein pocket.

Example usage for  `example/7x79_pocket10.pdb` and `example/7x79_ligand.sdf` :

```bash
python utils/smiles_conformer_generator.py --smiles {the_smiles_string} --pdb_path example/7x79_pocket10.pdb --sdf_path `example/7x79_ligand.sdf`
```

Running the above code will generate a sdf file in containing a docked conformer of the given smiles in the same folder as the given pocket. 

You can also process a batch of smiles by providing them in a csv format:
```bash
python utils/smiles_conformer_generator.py --csv_path {path_to_csv} --pdb_path {path_to_protein_pocket} --sdf_path {path_to_reference_ligand}
```

## Sample

Given the generated conformer using `utils/smiles_conformer_generator.py` (or any othe technique), the `template_based_sample_for_pdb.py` script can be used for fragment growning. 

```bash
python template_based_sample_for_pdb.py --pdb_path {path_to_protein_pdb} --sdf_path {path_to_fragment_structure} 
```



<!-- ## Datasets

Please refer to [`README.md`](./data/README.md) in the `data` folder.

## Sampling

**NOTE: It is highly recommended to add `taskset -c` to use only one cpu when sampling (e.g. `taskset -c 0 python sample_xxx.py` to use CPU 0), which is much faster. The reason is not clear yet.**

### Sampling for pockets in the testset

To sample molecules for the i-th pocket in the testset, please first download the trained models following [`README.md`](./ckpt/README.md) in the `ckpt` folder. 
Then, run the following command:

```bash
python sample.py --data_id {i} --outdir ./outputs  # Replace {i} with the index of the data. i should be between 0 and 99 for the testset.
```

We recommend to specify the GPU device number and restrict the cpu cores using command like:

```bash
CUDA_VISIBLE_DIVICES=0  taskset -c 0 python sample.py --data_id 0 --outdir ./outputs
```
We also provide a bash file `batch_sample.sh` for sampling molecules for the whole test set in parallel. For example, to sample with three workers, run the following commands in three panes.
```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0 bash batch_sample.sh  3 0 0

CUDA_VISIBLE_DEVICES=0 taskset -c 1 bash batch_sample.sh  3 1 0

CUDA_VISIBLE_DEVICES=0 taskset -c 2 bash batch_sample.sh  3 2 0
```
The three parameters of `batch_sample.py` represent the number of workers, the index of current worker and the start index of the datapoint in the test set, respectively.

**NOTE: We find it much faster to use only one CPU for one sampling program (i.e., set `taskset -c` to use one CPU).**

### Sampling for PDB pockets 
To generate ligands for your own pocket, you need to provide the `PDB` structure file of the protein, the center coordinate of the pocket bounding box, and optionally the side length of the bounding box (default: 23Å). Note that there is a blank before the first value of the `center` parameter. The blank cannot be omitted if the first value is negative (e.g., `--center  " -1.5,28.0,36.0"`).

Example:

```bash
python sample_for_pdb.py \
      --pdb_path ./example/4yhj.pdb
      --center " 32.0,28.0,36.0"
```

<img src="./assets/bounding_box.png" alt="bounding box" width="70%" />


## Training

```
python train.py --config ./configs/train.yml --logdir ./logs
```
For training, we recommend to install [`apex` ](https://github.com/NVIDIA/apex) for lower gpu memory usage. If  so, change the value of `train/use_apex` in the `configs/train.yml` file.

## Citation
```
@inproceedings{peng2022pocket2mol,
  title={Pocket2Mol: Efficient Molecular Sampling Based on 3D Protein Pockets},
  author={Xingang Peng and Shitong Luo and Jiaqi Guan and Qi Xie and Jian Peng and Jianzhu Ma},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```

## Contact 
Xingang Peng (xingang.peng@gmail.com) -->
