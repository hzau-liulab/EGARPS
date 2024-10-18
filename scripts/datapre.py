import argparse
import copy
import logging
import math
import os
import time
import pickle
import random
import shutil
import subprocess
from itertools import combinations
from itertools import groupby
from typing import Any, Dict, Union
from typing import List, Optional, Tuple
from parallel import submit_jobs
import atom3.database as db
import dgl
import numpy as np
import pandas as pd
import scipy.spatial as spatial
import torch
from Bio import SeqIO
from Bio.PDB import PDBParser, get_surface, internal_coords, NACCESS
from Bio.PDB.ResidueDepth import min_dist
from biopandas.pdb import PandasPdb
from dgl import DGLError
from numpy import linalg as LA
from pandas import Index
from torch import Tensor
import pymol
import io
import tempfile
from Bio.PDB.PDBIO import PDBIO
from scipy.spatial import distance as scidist
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

from featuralize import featuralize

def mass_and_charge(pdb):
    feats=featuralize(pdb)
    return torch.tensor(feats,dtype=torch.float32)

# # =============================================================================
# # ## from project.utils.set.runtime.utils import update_relative_positions, \
# #     update_potential_values, find_intersection_indices_2D
# # =============================================================================
import ctypes
from functools import wraps
import torch.distributed as dist
from torch import FloatTensor

def validate_residues(residues: List[pd.DataFrame], input_file: str) -> List[pd.DataFrame]:
    """Ensure each residue has a valid nitrogen (N), carbon-alpha (Ca), and carbon (C) atom."""
    residues_filtered = []
    for residue_idx, residue in enumerate(residues):
        df = residue[1]
        n_atom = df[df['atom_name'] == 'N']
        ca_atom = df[df['atom_name'] == 'CA']
        c_atom = df[df['atom_name'] == 'C']
        if n_atom.shape[0] == 1 and ca_atom.shape[0] == 1 and c_atom.shape[0] == 1:
            residues_filtered.append(residue)
        else:
            raise Exception(f'Residue {residue_idx} in {input_file} did not contain at least one valid backbone atom.')
    return residues_filtered


def build_geometric_vectors(residues: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive all residue-local geometric vector quantities."""
    n_i_list = []
    u_i_list = []
    v_i_list = []
    residue_representatives_loc_list = []

    for residue in residues:
        df = residue[1]
        
        if all(x in df['atom_name'].values for x in ['N','CA','C']):
            n_atom = df[df['atom_name'] == 'N']
            ca_atom = df[df['atom_name'] == 'CA']
            c_atom = df[df['atom_name'] == 'C']
        elif all(x in df['atom_name'].values for x in ["C5'","C3'","O5'"]):
            n_atom = df[df['atom_name'] == "C5'"]
            ca_atom = df[df['atom_name'] == "C3'"]
            c_atom = df[df['atom_name'] == "O5'"]
        else:
            raise Exception('Check residue atoms')

        if n_atom.shape[0] != 1 or ca_atom.shape[0] != 1 or c_atom.shape[0] != 1:
            logging.info(df.iloc[0, :])
            print(df.iloc[0,:])
            raise ValueError('In compute_rel_geom_feats(), no N/CA/C exists')

        n_loc = n_atom[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
        ca_loc = ca_atom[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
        c_loc = c_atom[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)

        u_i = (n_loc - ca_loc) / LA.norm(n_loc - ca_loc)
        t_i = (c_loc - ca_loc) / LA.norm(c_loc - ca_loc)
        n_i = np.cross(u_i, t_i) / LA.norm(np.cross(u_i, t_i))
        v_i = np.cross(n_i, u_i)
        assert (math.fabs(LA.norm(v_i) - 1.) < 1e-5), 'In compute_rel_geom_feats(), v_i norm was larger than 1'

        n_i_list.append(n_i)
        u_i_list.append(u_i)
        v_i_list.append(v_i)

        residue_representatives_loc_list.append(ca_loc)

    residue_representatives_loc_feat = np.stack(residue_representatives_loc_list, axis=0)  # (num_res, 3)
    n_i_feat = np.stack(n_i_list, axis=0)
    u_i_feat = np.stack(u_i_list, axis=0)
    v_i_feat = np.stack(v_i_list, axis=0)

    return residue_representatives_loc_feat, n_i_feat, u_i_feat, v_i_feat


def compute_rel_geom_feats(graph: dgl.DGLGraph,
                            atoms_df: pd.DataFrame,
                            input_file='Structure.pdb',
                            check_residues=False) -> torch.Tensor:
    """Calculate the relative geometric features for each residue's local coordinate system."""
    # Collect all residues along with their constituent atoms
    residues = list(atoms_df.groupby(['chain_id', 'residue_number', 'insertion']))  # Note: Not the sequence order!

    # Validate the atom-wise composition of each residue
    if check_residues:
        residues = validate_residues(residues, input_file)

    # Derive zero-based node-wise residue numbers
    residue_numbers = graph.ndata['residue_number'] - 1

    # Derive all geometric vector quantities specific to each residue
    residue_representatives_loc_feat, n_i_feat, u_i_feat, v_i_feat = build_geometric_vectors(residues)

    # Loop over all edges of the graph, and build the various p_ij, q_ij, k_ij, t_ij pairs
    edges = graph.edges()
    edge_feat_ori_list = []
    for edge in zip(edges[0], edges[1]):
        # Get edge metadata
        src = edge[0]
        dst = edge[1]
        res_src = residue_numbers[src]
        res_dst = residue_numbers[dst]

        # Place n_i, u_i, v_i as lines in a 3 x 3 basis matrix
        basis_matrix = np.stack((n_i_feat[res_dst, :], u_i_feat[res_dst, :], v_i_feat[res_dst, :]), axis=0)
        p_ij = np.matmul(basis_matrix,
                          residue_representatives_loc_feat[res_src, :] -
                          residue_representatives_loc_feat[res_dst, :])
        q_ij = np.matmul(basis_matrix, n_i_feat[res_src, :])  # Shape: (3,)
        k_ij = np.matmul(basis_matrix, u_i_feat[res_src, :])
        t_ij = np.matmul(basis_matrix, v_i_feat[res_src, :])
        s_ij = np.concatenate((p_ij, q_ij, k_ij, t_ij), axis=0)  # Shape: (12,)
        edge_feat_ori_list.append(s_ij)

    # Return our resulting relative geometric features, local to each residue
    edge_feat_ori_feat = torch.from_numpy(np.stack(edge_feat_ori_list, axis=0)).float().to(graph.device)  # (n_edge, 12)
    return edge_feat_ori_feat


def min_max_normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize provided tensor to have values be in range [0, 1]."""
    # Credit: https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122
    tensor -= tensor.min()
    tensor /= tensor.max()
    return tensor


def compute_euclidean_distance_matrix(points: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between every input point (i.e., row).

    Parameters
    ----------
    points: torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    euclidean_dists = torch.norm(points[:, None] - points, dim=2, p=2)
    return euclidean_dists


def find_intersection_indices_2D(first: torch.Tensor, second: torch.Tensor, device: str) -> torch.Tensor:
    """
    Return the row indices (in the first input Tensor) also present in the second input Tensor using NumPy.

    Parameters
    ----------
    first: torch.Tensor
        Primary input tensor.
    second:
        Secondary input tensor.
    device:
        Device on which to put index Tensor indicating rows in first that were also in second.

    Returns
    -------
    torch.Tensor
        A Tensor of row indices shared by both input Tensors.
    """
    first_np = first.cpu().numpy()
    second_np = second.cpu().numpy()
    first_np_view = first_np.view([('', first_np.dtype)] * first_np.shape[1])
    second_np_view = second_np.view([('', second_np.dtype)] * second_np.shape[1])
    intersect_idx = torch.tensor(np.intersect1d(first_np_view, second_np_view, return_indices=True)[1], device=device)
    return intersect_idx


def compute_bond_edge_feats(graph: dgl.DGLGraph, first_iter=False) -> torch.Tensor:
    """
    Compute edge feature indicating whether a covalent bond exists between a pair of atoms.

    Parameters
    ----------
    graph: dgl.DGLGraph
    first_iter: bool

    Returns
    -------
    torch.Tensor
    """
    # Compute all atom-atom Euclidean distances as a single distance matrix
    coords_distance_matrix = compute_euclidean_distance_matrix(graph.ndata['x_pred'])

    # Create a covalent 'distance' matrix by adding the radius array with its transpose
    orig_covalent_radius_distance_matrix = torch.add(
        graph.ndata['covalent_radius'].reshape(-1, 1),
        graph.ndata['covalent_radius'].reshape(1, -1)
    )

    # Add the covalent bond distance tolerance to the original covalent radius distance matrix
    covalent_radius_distance_matrix = (orig_covalent_radius_distance_matrix + COVALENT_RADIUS_TOLERANCE)

    # Sanity-check values in both distance matrices, only when first computing covalent bonds
    if first_iter:
        assert not torch.isnan(coords_distance_matrix).any(), 'No NaNs are allowed as coordinate pair distances'
        assert not torch.isnan(covalent_radius_distance_matrix).any(), 'No NaNs are allowed as covalent distances'

    # Threshold distance matrix to entries where Euclidean distance D > 0.4 and D < (covalent radius + tolerance)
    coords_distance_matrix[coords_distance_matrix <= 0.4] = torch.nan
    coords_distance_matrix[coords_distance_matrix >= covalent_radius_distance_matrix] = torch.nan
    covalent_bond_matrix = torch.nan_to_num(coords_distance_matrix)
    covalent_bond_matrix[covalent_bond_matrix > 0] = 1

    # Derive relevant covalent bonds based on the binary covalent bond matrix computed previously
    graph_edges_with_eids = graph.edges(form='all')
    graph_edges = torch.cat(
        (graph_edges_with_eids[0].reshape(-1, 1),
          graph_edges_with_eids[1].reshape(-1, 1)),
        dim=1
    )
    covalent_bond_edge_indices = covalent_bond_matrix.nonzero()
    combined_edges = torch.cat((graph_edges, covalent_bond_edge_indices))
    unique_edges, edge_counts = combined_edges.unique(dim=0, return_counts=True)
    covalently_bonded_edges = unique_edges[edge_counts > 1]

    # Find edges in the original graph for which a covalent bond was discovered
    covalently_bonded_eids = find_intersection_indices_2D(graph_edges, covalently_bonded_edges, graph_edges.device)

    # Craft new bond features based on the covalent bonds discovered above
    covalent_bond_feats = torch.zeros((len(graph_edges), 1), device=graph_edges.device)
    covalent_bond_feats[covalently_bonded_eids] = 1.0

    return covalent_bond_feats


def compute_chain_matches(edges: dgl.udf.EdgeBatch) -> Dict:
    """
    Compute edge feature indicating whether a pair of atoms belong to the same chain.

    Parameters
    ----------
    edges: dgl.udf.EdgeBatch

    Returns
    -------
    dict
    """
    # Collect edges' source and destination node IDs, as well as all nodes' chain IDs
    in_same_chain = torch.eq(edges.src['chain_id'], edges.dst['chain_id']).long().float()
    return {'in_same_chain': in_same_chain}


def compute_molecular_matches(edges):
    src_moltype=edges.src['moltype']
    dst_moltype=edges.dst['moltype']
    in_same_molecular=torch.zeros(src_moltype.shape[0],3)
    for idt,i,j in zip(range(src_moltype.shape[0]),src_moltype,dst_moltype):
        if i==j==0:
            in_same_molecular[idt,0]=1
        elif i==j==1:
            in_same_molecular[idt,1]=1
        else:
            in_same_molecular[idt,2]=1
    return {'in_same_molecular':in_same_molecular.float()}


def get_r(graph: dgl.DGLGraph):
    """Compute inter-nodal distances"""
    cloned_rel_pos = torch.clone(graph.edata['rel_pos'])
    if graph.edata['rel_pos'].requires_grad:
        cloned_rel_pos.requires_grad_()
    return torch.sqrt(torch.sum(cloned_rel_pos ** 2, -1, keepdim=True))


def apply_potential_function(edge: dgl.udf.EdgeBatch):
    potential_parameters = torch.cosine_similarity(edge.src['surf_prox'], edge.dst['surf_prox']).float().reshape(-1, 1)
    return {'w': potential_function(edge.data['r'], potential_parameters)}


def potential_function(r: Tensor, potential_parameters: FloatTensor):
    x = r - potential_parameters - 1
    potential_function_global_min = -0.321919
    return x ** 4 - x ** 2 + 0.1 * x - potential_function_global_min


def potential_gradient(r: Tensor, potential_parameters: FloatTensor):
    x = r - potential_parameters - 1
    return 4 * x ** 3 - 2 * x + 0.1


def iset_copy_without_weak_connections(orig_graph: dgl.DGLGraph,
                                        graph_idx: Tensor,
                                        edges_per_node: int,
                                        batched_input: bool,
                                        pdb_filepaths: List[str]):
    """Make a copy of a graph, preserving only the edges_per_node-strongest incoming edges for each node.

    Args
        orig_graph: dgl.DGLGraph, the graph to copy
        graph_idx: Tensor, the indices of subgraphs (in a batched graph) to copy
        edges_per_node: int, the number of connections to preserve for each node in the output graph
        batched_input: bool, whether the input graph is a batched graph, comprised of multiple subgraphs
        pdb_filepaths: List[str], a collection of paths to input PDB files
    """
    # Cache the original batch number of nodes and edges
    batch_num_nodes, batch_num_edges = None, None
    if batched_input:
        batch_num_nodes = orig_graph.batch_num_nodes()
        batch_num_edges = orig_graph.batch_num_edges()

    # Iterate through all batched subgraphs to be rewired, since not all subgraphs have to have the same number of nodes
    graphs = dgl.unbatch(orig_graph) if batched_input else [orig_graph]
    for i in graph_idx.reshape(-1, 1):
        # Gather graph data
        graph = graphs[i.squeeze()]
        num_nodes = graph.num_nodes()
        w = graph.edata['w']
        srcs, dsts = graph.all_edges()

        # Load input PDB as a DataFrame
        atoms_df = PandasPdb().read_pdb(pdb_filepaths[i]).df['ATOM']

        # Sort srcs, dsts and w by dsts_node_id, then srcs_node_id
        w = w.view(num_nodes, edges_per_node)  # [dsts, srcs]
        dsts = dsts.view(num_nodes, edges_per_node)
        srcs = srcs.view(num_nodes, edges_per_node)

        # Sort edges according to their weight
        w, indices = torch.sort(w, descending=True)
        dsts = torch.gather(dsts, dim=-1, index=indices)
        srcs = torch.gather(srcs, dim=-1, index=indices)

        # Take the top-edges_per_node edges
        num_edges = graph.num_nodes() * edges_per_node
        dsts = dsts[:, :num_edges]
        srcs = srcs[:, :num_edges]

        # Reshape into 1D
        dsts = torch.reshape(dsts, (num_nodes * edges_per_node,)).detach()
        srcs = torch.reshape(srcs, (num_nodes * edges_per_node,)).detach()

        # Create new graph with fewer edges and fill with data
        rewired_graph = dgl.graph(data=(srcs, dsts), num_nodes=graph.num_nodes())
        # Fill in node data
        rewired_graph.ndata['f'] = graph.ndata['f']
        rewired_graph.ndata['chain_id'] = graph.ndata['chain_id']
        rewired_graph.ndata['x_pred'] = graph.ndata['x_pred']
        rewired_graph.ndata['x_true'] = graph.ndata['x_true']
        rewired_graph.ndata['labeled'] = graph.ndata['labeled']
        rewired_graph.ndata['interfacing'] = graph.ndata['interfacing']
        rewired_graph.ndata['covalent_radius'] = graph.ndata['covalent_radius']
        rewired_graph.ndata['residue_number'] = graph.ndata['residue_number']
        rewired_graph.ndata['is_ca_atom'] = graph.ndata['is_ca_atom']
        # Fill in edge data
        edge_dtype = graph.edata['edge_dist'].dtype
        edge_dist = torch.norm(rewired_graph.ndata['x_pred'][dsts] - rewired_graph.ndata['x_pred'][srcs], dim=1, p=2)
        rewired_graph.edata['edge_dist'] = min_max_normalize_tensor(edge_dist).reshape(-1, 1)
        rewired_graph.edata['edge_dist'] = rewired_graph.edata['edge_dist'].type(edge_dtype)
        rewired_graph.edata['bond_type'] = compute_bond_edge_feats(rewired_graph)
        rewired_graph.edata['bond_type'] = rewired_graph.edata['bond_type'].type(edge_dtype)
        rewired_graph.apply_edges(compute_chain_matches)  # Install edata['in_same_chain']
        rewired_graph.edata['in_same_chain'] = rewired_graph.edata['in_same_chain'].type(edge_dtype)
        rewired_graph.edata['rel_geom_feats'] = compute_rel_geom_feats(rewired_graph, atoms_df)
        rewired_graph.edata['rel_geom_feats'] = rewired_graph.edata['rel_geom_feats'].type(edge_dtype)
        # Combine individual edge features into a single edge feature tensor
        rewired_graph.edata['f'] = torch.cat((
            rewired_graph.edata['edge_dist'],
            rewired_graph.edata['bond_type'],
            rewired_graph.edata['in_same_chain'],
            rewired_graph.edata['rel_geom_feats']
        ), dim=1)
        rewired_graph.edata['f'] = rewired_graph.edata['f'].type(edge_dtype)

        # Use newly-selected edges to update relative positions and potentials between nodes j and nodes i
        update_relative_positions(rewired_graph)
        update_potential_values(rewired_graph, r=rewired_graph.edata['r'])

        # Update subgraph in original list of batched graphs
        graphs[i] = rewired_graph

    # Re-batch graphs after in-place updating those that were rewired
    orig_graph = dgl.batch(graphs) if batched_input else graphs[0]

    # Restore the original batch number of nodes and edges
    if batched_input:
        orig_graph.set_batch_num_nodes(batch_num_nodes)
        orig_graph.set_batch_num_edges(batch_num_edges)

    return orig_graph


def update_absolute_positions(graph: dgl.DGLGraph, pos_updates: Tensor, absolute_position_key='x_pred'):
    """For each node in the graph, update the absolute position of the corresponding node.
      Write the updated absolute positions to the graph as node data."""
    graph.ndata[absolute_position_key] = graph.ndata[absolute_position_key] + pos_updates


def update_relative_positions(graph: dgl.DGLGraph, relative_position_key='rel_pos', absolute_position_key='x_pred'):
    """For each directed edge in the graph, calculate the relative position of the destination node with respect
    to the source node. Write the relative positions to the graph as edge data."""
    srcs, dsts = graph.all_edges()
    absolute_positions = graph.ndata[absolute_position_key]
    graph.edata[relative_position_key] = absolute_positions[dsts] - absolute_positions[srcs]
    graph.edata['r'] = graph.edata[relative_position_key].norm(dim=-1, keepdim=True)


def update_potential_values(graph: dgl.DGLGraph, r=None):
    """For each directed edge in the graph, compute the value of the potential between source and destination nodes.
    Write the computed potential values to the graph as edge data."""
    if r is None:
        r = get_r(graph)
    graph.edata['r'] = r
    graph.apply_edges(func=apply_potential_function)


def aggregate_residual(feats1, feats2, method: str):
    """ Add or concatenate two fiber features together. If degrees don't match, will use the ones of feats2. """
    if method in ['add', 'sum']:
        return {k: (v + feats1[k]) if k in feats1 else v for k, v in feats2.items()}
    elif method in ['cat', 'concat']:
        return {k: torch.cat([v, feats1[k]], dim=1) if k in feats1 else v for k, v in feats2.items()}
    else:
        raise ValueError('Method must be add/sum or cat/concat')


def degree_to_dim(degree: int) -> int:
    return 2 * degree + 1


def unfuse_features(features: Tensor, degrees: List[int]) -> Dict[str, Tensor]:
    return dict(zip(map(str, degrees), features.split([degree_to_dim(deg) for deg in degrees], dim=-1)))


def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def to_cuda(x):
    """ Try to convert a Tensor, a collection of Tensors or a DGLGraph to CUDA """
    if isinstance(x, Tensor):
        return x.cuda(non_blocking=True)
    elif isinstance(x, tuple):
        return (to_cuda(v) for v in x)
    elif isinstance(x, list):
        return [to_cuda(v) for v in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        # DGLGraph or other objects
        return x.to(device=torch.cuda.current_device())


def get_local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', 0))


def init_distributed() -> bool:
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1
    if distributed:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
        if backend == 'nccl':
            torch.cuda.set_device(get_local_rank())
        else:
            logging.warning('Running on CPU only!')
        assert torch.distributed.is_initialized()
    return distributed


def increase_l2_fetch_granularity():
    # maximum fetch granularity of L2: 128 bytes
    _libcudart = ctypes.CDLL('libcudart.so')
    # set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank_zero_only(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


def using_tensor_cores(amp: bool) -> bool:
    major_cc, minor_cc = torch.cuda.get_device_capability()
    return (amp and major_cc >= 7) or major_cc >= 8

RESI_THREE_TO_1: Dict[str, str] = {
    "ALA": "A",
    "ASX": "B",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "GLX": "Z",
    "CSD": "C",
    "HYP": "P",
    "BMT": "T",
    "3HP": "X",
    "4HP": "X",
    "5HP": "Q",
    "ACE": "X",
    "ABA": "A",
    "AIB": "A",
    "NH2": "X",
    "CBX": "X",
    "CSW": "C",
    "OCS": "C",
    "DAL": "A",
    "DAR": "R",
    "DSG": "N",
    "DSP": "D",
    "DCY": "C",
    "CRO": "TYG",
    "DGL": "E",
    "DGN": "Q",
    "DHI": "H",
    "DIL": "I",
    "DIV": "V",
    "DLE": "L",
    "DLY": "K",
    "DPN": "F",
    "DPR": "P",
    "DSN": "S",
    "DTH": "T",
    "DTR": "W",
    "DTY": "Y",
    "DVA": "V",
    "FOR": "X",
    "CGU": "E",
    "IVA": "X",
    "KCX": "K",
    "LLP": "K",
    "CXM": "M",
    "FME": "M",
    "MLE": "L",
    "MVA": "V",
    "NLE": "L",
    "PTR": "Y",
    "ORN": "A",
    "SEP": "S",
    "SEC": "U",
    "TPO": "T",
    "PCA": "Q",
    "PVL": "X",
    "PYL": "O",
    "SAR": "G",
    "CEA": "C",
    "CSO": "C",
    "CSS": "C",
    "CSX": "C",
    "CME": "C",
    "TYS": "Y",
    "BOC": "X",
    "TPQ": "Y",
    "STY": "Y",
    "UNK": "X",
}
"""
Mapping of 3-letter residue names to 1-letter residue names.
Non-standard/modified amino acids are mapped to their parent amino acid.
Includes ``"UNK"`` to denote unknown residues.
"""

BASE_AMINO_ACIDS: List[str] = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
"""Vocabulary of 20 standard amino acids."""

# Atom classes based on Heyrovska, Raji covalent radii paper.
DEFAULT_BOND_STATE: Dict[str, str] = {"N": "Nsb", "CA": "Csb", "C": "Cdb", "O": "Odb", "OXT": "Osb", "CB": "Csb",
                                      "H": "Hsb",
                                      # Not sure about these - assuming they're all standard Hydrogen. Won't make much difference given
                                      # the tolerance is larger than Hs covalent radius
                                      "HG1": "Hsb", "HE": "Hsb", "1HH1": "Hsb", "1HH2": "Hsb", "2HH1": "Hsb",
                                      "2HH2": "Hsb", "HG": "Hsb", "HH": "Hsb", "1HD2": "Hsb", "2HD2": "Hsb",
                                      "HZ1": "Hsb", "HZ2": "Hsb", "HZ3": "Hsb", }

RESIDUE_ATOM_BOND_STATE: Dict[str, Dict[str, str]] = {
    "XXX": {"N": "Nsb", "CA": "Csb", "C": "Cdb", "O": "Odb", "OXT": "Osb", "CB": "Csb", "H": "Hsb", },
    "VAL": {"CG1": "Csb", "CG2": "Csb"}, "LEU": {"CG": "Csb", "CD1": "Csb", "CD2": "Csb"},
    "ILE": {"CG1": "Csb", "CG2": "Csb", "CD1": "Csb"}, "MET": {"CG": "Csb", "SD": "Ssb", "CE": "Csb"},
    "PHE": {"CG": "Cdb", "CD1": "Cres", "CD2": "Cres", "CE1": "Cdb", "CE2": "Cdb", "CZ": "Cres", },
    "PRO": {"CG": "Csb", "CD": "Csb"}, "SER": {"OG": "Osb"}, "THR": {"OG1": "Osb", "CG2": "Csb"}, "CYS": {"SG": "Ssb"},
    "ASN": {"CG": "Csb", "OD1": "Odb", "ND2": "Ndb"}, "GLN": {"CG": "Csb", "CD": "Csb", "OE1": "Odb", "NE2": "Ndb"},
    "TYR": {"CG": "Cdb", "CD1": "Cres", "CD2": "Cres", "CE1": "Cdb", "CE2": "Cdb", "CZ": "Cres", "OH": "Osb", },
    "TRP": {"CG": "Cdb", "CD1": "Cdb", "CD2": "Cres", "NE1": "Nsb", "CE2": "Cdb", "CE3": "Cdb", "CZ2": "Cres",
            "CZ3": "Cres", "CH2": "Cdb", }, "ASP": {"CG": "Csb", "OD1": "Ores", "OD2": "Ores"},
    "GLU": {"CG": "Csb", "CD": "Csb", "OE1": "Ores", "OE2": "Ores"},
    "HIS": {"CG": "Cdb", "CD2": "Cdb", "ND1": "Nsb", "CE1": "Cdb", "NE2": "Ndb", },
    "LYS": {"CG": "Csb", "CD": "Csb", "CE": "Csb", "NZ": "Nsb"},
    "ARG": {"CG": "Csb", "CD": "Csb", "NE": "Nsb", "CZ": "Cdb", "NH1": "Nres", "NH2": "Nres", }, }

# Covalent radii from Heyrovska, Raji : 'Atomic Structures of all the Twenty
# Essential Amino Acids and a Tripeptide, with Bond Lengths as Sums of Atomic
# Covalent Radii' <https://arxiv.org/pdf/0804.2488.pdf>
# Adding Ores between Osb and Odb for Asp and Glu, Nres between Nsb and Ndb
# for Arg, as PDB does not specify

COVALENT_RADII: Dict[str, float] = {"Csb": 0.77, "Cres": 0.72, "Cdb": 0.67, "Osb": 0.67, "Ores": 0.635, "Odb": 0.60,
                                    "Nsb": 0.70, "Nres": 0.66, "Ndb": 0.62, "Hsb": 0.37, "Ssb": 1.04, }

COVALENT_RADIUS_TOLERANCE = 0.56  # 0.4, 0.45, or 0.56 - These are common distance tolerances for covalent bonds

# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepRefine (https://github.com/BioinfoMachineLearning/DeepRefine):
# -------------------------------------------------------------------------------------------------------------------------------------
PROT_ATOM_NAMES = ['N', 'CA', 'C', 'O', 'CB', 'OG', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OD1', 'ND2', 'CG1', 'CG2',
                    'CD', 'CE', 'NZ', 'OD2', 'OE1', 'NE2', 'OE2', 'OH', 'NE', 'NH1', 'NH2', 'OG1', 'SD', 'ND1', 'SG',
                    'NE1', 'CE3', 'CZ2', 'CZ3', 'CH2', 'OXT', 'UNX']  # 'UNX' represents the unknown atom type

STD_ATOM_NAMES = [
    'CA', 'N', 'C', 'O', 'CB', 'CG', 'CD2', 'CD1', 'CG1', 'CG2', 'CD',
    'OE1', 'OE2', 'OG', 'OG1', 'OD1', 'OD2', 'CE', 'NZ', 'NE', 'CZ',
    'NH2', 'NH1', 'ND2', 'CE2', 'CE1', 'NE2', 'OH', 'ND1', 'SD', 'SG',
    'NE1', 'CE3', 'CZ3', 'CZ2', 'CH2', 'P', "C3'", "C4'", "O3'", "C5'",
    "O5'", "O4'", "C1'", "C2'", "O2'", 'OP1', 'OP2', 'N9', 'N2', 'O6',
    'N7', 'C8', 'N1', 'N3', 'C2', 'C4', 'C6', 'C5', 'N6', 'N4', 'O2',
    'O4', 'OXT', 'UNX'
]

# Dataset-global maximum constant for feature normalization
MAX_FEATS_CONST = 100.0

# Dihedral angle mapping
DIHEDRAL_ANGLE_ID_TO_ATOM_NAMES = {
    0: ['N', 'CA', 'C', 'N'],
    1: ['CA', 'C', 'N', 'CA'],
    2: ['C', 'N', 'CA', 'C']
}

ENCODE_RES = ['ALA','ARG','ASN','ASP','CYS',
              'GLN','GLU','GLY','HIS','ILE',
              'LEU','LYS','MET','PHE','PRO',
              'SER','THR','TRP','TYR','VAL',
              'A','G','C','U']

def get_allowable_feats(ca_only: bool):
    return [BASE_AMINO_ACIDS, ] if ca_only else [STD_ATOM_NAMES, ]

def reindex_df_field_values(df: pd.DataFrame, field_name: str, start_index: int) -> pd.DataFrame:
    """
    Reindex a Series of consecutive integers, corresponding to a specific DataFrame field, to start from a given index.

    Parameters
    ----------
    df: pd.DataFrame
    field_name: str
    start_index: int

    Returns
    -------
    pd.DataFrame
    """
    if field_name == 'residue_number':
        field_values = df[['residue_number','insertion']].values.squeeze().tolist()
    else:
        field_values = df[[field_name]].values.squeeze().tolist()
    reindexed_field_values = [c for c, (k, g) in enumerate(groupby(field_values), start_index) for _ in g]
    df[[field_name]] = np.array(reindexed_field_values).reshape(-1, 1)  # Install reindexed field values
    return df

def get_shared_df_coords(left_df: pd.DataFrame, right_df: pd.DataFrame, merge_order: str, return_np_coords=False) \
        -> Union[Tuple[Union[torch.Tensor, np.ndarray], pd.Index], pd.Index]:
    """Reconcile two dataframes to get the 3D coordinates corresponding to each atom shared by both input DataFrames."""
    on_cols = ['atom_name', 'chain_id', 'residue_number']
    if merge_order == 'left':
        merged_df = left_df.merge(right_df, how='inner', on=on_cols)
        orig_left_df_indices = retrieve_indices_in_orig_df(merged_df, left_df)  # Retrieve orig. IDs for merged rows
        if return_np_coords:
            true_coords = merged_df[['x_coord_y', 'y_coord_y', 'z_coord_y']].to_numpy()
        else:
            true_coords = torch.tensor(merged_df[['x_coord_y', 'y_coord_y', 'z_coord_y']].values, dtype=torch.float32)
        return true_coords, orig_left_df_indices
    elif merge_order == 'right':
        merged_df = right_df.merge(left_df, how='inner', on=on_cols)
        orig_right_df_indices = retrieve_indices_in_orig_df(merged_df, right_df)  # Retrieve orig. IDs for merged rows
        return orig_right_df_indices
    else:
        raise NotImplementedError(f'Merge order {merge_order} is not currently supported')

def retrieve_indices_in_orig_df(merged_df: pd.DataFrame, orig_df: pd.DataFrame) -> pd.Index:
    """Get the original indices corresponding to the merged DataFrame's rows."""
    on_cols = ['atom_name', 'chain_id', 'residue_number']
    unique_merged_cols_array = merged_df.drop(labels=merged_df.columns.difference(on_cols), axis=1)
    matched_df = orig_df.reset_index().merge(right=unique_merged_cols_array, how='inner', on=on_cols).set_index('index')
    return matched_df.index

def get_interfacing_atom_indices(true_atom_df: pd.DataFrame,
                                  orig_true_df_indices: pd.Index,
                                  idt: float,
                                  return_partners=False) -> Tuple[Index, Optional[Dict[Any, Any]]]:
    """
    For a given true PDB DataFrame, return the indices of atoms found in any interface between at least two chains.

    Parameters
    ----------
    true_atom_df: pd.DataFrame
    orig_true_df_indices: pd.Index
    idt: float
    return_partners: bool

    Returns
    -------
    Tuple[Index, Union[Dict[int, Set[Any]], Index]]
    """
    # Filter down to only the atoms contained within the predicted PDB structure, and clean up these atoms' chain IDs
    atoms = true_atom_df.loc[orig_true_df_indices, :]
    atoms.reset_index(drop=True,
                      inplace=True)  # Make selected atoms' indices start from zero for following computations
    atoms = reindex_df_field_values(atoms, field_name='chain_id', start_index=0)
    unique_chain_ids = atoms['chain_id'].unique().tolist()
    unique_chain_ids.sort()
    # Pre-compute all pairwise atom distances
    all_atom_coords = atoms[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    atom_coord_tree = spatial.cKDTree(all_atom_coords)
    # Find the index of all inter-chain atoms satisfying the specified interface distance (IDT) threshold
    inter_chain_atom_index_mapping = {
        i: atoms[atoms['chain_id'] != chain_id].index.values.tolist()
        for i, chain_id in enumerate(unique_chain_ids)
    }
    interfacing_atom_indices = set()
    interfacing_atom_mapping = {} if return_partners else None
    for i, atom in enumerate(atoms.itertuples(index=False)):
        inter_chain_atom_indices = inter_chain_atom_index_mapping[atom.chain_id]
        atom_neighbor_indices = atom_coord_tree.query_ball_point([atom.x_coord, atom.y_coord, atom.z_coord], idt)
        inter_chain_atom_neighbor_indices = set(atom_neighbor_indices) & set(inter_chain_atom_indices)
        interfacing_atom_indices = interfacing_atom_indices.union(inter_chain_atom_neighbor_indices)
        if return_partners:
            interfacing_atom_mapping[i] = list(inter_chain_atom_neighbor_indices)
            interfacing_atom_mapping[i].sort()
    # Sort collected atom indices and return them as a Pandas Index
    interfacing_atom_indices = list(interfacing_atom_indices)
    interfacing_atom_indices.sort()
    interfacing_atom_indices = pd.Index(interfacing_atom_indices)
    return interfacing_atom_indices, interfacing_atom_mapping

def assign_bond_states_to_atom_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a PandasPDB atom dataframe and assign bond states to each atom based on:
    Atomic Structures of all the Twenty Essential Amino Acids and a Tripeptide, with Bond Lengths as Sums of Atomic Covalent Radii
    Heyrovska, 2008

    :param df: Pandas PDB dataframe
    :type df: pd.DataFrame
    :return: Dataframe with added atom_bond_state column
    :rtype: pd.DataFrame
    """
    # Map atoms to their standard bond states
    naive_bond_states = pd.Series(df["atom_name"].map(DEFAULT_BOND_STATE))

    # Create series of bond states for non-standard states
    ss = (
        pd.DataFrame(RESIDUE_ATOM_BOND_STATE)
            .unstack()
            .rename_axis(("residue_name", "atom_name"))
            .rename("atom_bond_state")
    )

    # Map non-standard states to the dataframe based on residue name-atom name pairs
    df = df.join(ss, on=["residue_name", "atom_name"])

    # Fill all NaNs with standard states
    df = df.fillna(value={"atom_bond_state": naive_bond_states})

    # Note: For example, in the case of ligand input, replace remaining NaN values with the most common bond state
    if df["atom_bond_state"].isna().sum() > 1:
        most_common_bond_state = df["atom_bond_state"].value_counts().index[0]
        df = df.fillna(value={"atom_bond_state": most_common_bond_state})

    return df

def max_normalize_array(array: np.ndarray, max_value: float = None) -> np.ndarray:
    """Normalize values in provided array using its maximum value."""
    if array.size > 0:
        array = array / (array.max() if max_value is None else max_value)
    return array

# def compute_surface_proximities(graph: dgl.DGLGraph, pdb_filepath: str, ca_only: bool) -> torch.Tensor:
#     """
#     For all nodes (i.e., atoms), find their proximity to their respective solvent-accessible surface area.
#     Such a scalar can describe how close to the chain surface a given atom is, which, as such,
#     can serve as a measure for an atom's "buriedness" (an important descriptor for tasks involving
#     multiple biological entities interacting with one another).

#     Parameters
#     ----------
#     graph: dgl.DGLGraph
#     pdb_filepath: str
#     ca_only: bool

#     Returns
#     -------
#     torch.Tensor
#     """
#     # If requested, compute residue buriedness using average vector ratio norms for each residue
#     if ca_only:
#         # return compute_mean_vector_ratio_norm(graph, pdb_filepath)
#         print('NNNNNNNNNNNNNNNNNNN')

#     # Extract from MSMS the vectors describing each chain's surface
#     parser = PDBParser()
#     pdb_code = db.get_pdb_code(pdb_filepath)
#     structure = parser.get_structure(pdb_code, pdb_filepath)
#     num_struct_models = len(structure)
#     assert num_struct_models == 1, f'Input PDB {pdb_filepath} must consist of only a single model'
#     model = structure[0]
#     chain_id_map = {k: c for c, (k, g) in enumerate(groupby([chain.id for chain in model]), 0) for _ in g}
#     surfaces = {chain_id_map[chain.id]: get_surface(chain) for chain in model}

#     # Derive the depth of each atom in the given graph
#     atom_depth_map = {v: [] for v in chain_id_map.values()}
#     for node_id in graph.nodes():
#         node_x = tuple(graph.ndata['x_pred'][node_id, :].numpy())
#         chain_id = graph.ndata['chain_id'][node_id].item()
#         atom_depth_map[chain_id].append(min_dist(node_x, surfaces[chain_id]))

#     # Normalize each chain's atom depths in batch yet separately from all other chain batches
#     surf_prox_list = []
#     for k in atom_depth_map.keys():
#         # Compile all atom depths together as a 2D NumPy array
#         atom_depths = np.array(atom_depth_map[k]).astype(np.float32).reshape(-1, 1)
#         # Normalize atom depths using a specified maximum value
#         atom_depths_scaled = max_normalize_array(atom_depths, max_value=100.0)
#         # Take the elementwise complement of atom depths to get the normalized surface proximity of each atom
#         surf_prox = np.ones_like(atom_depths_scaled) - atom_depths_scaled
#         # Ensure minimum surface proximity is zero, given that atoms may be more than 100 Angstrom away from a surface
#         clipped_surf_prox = surf_prox.clip(0.0, surf_prox.max())
#         surf_prox_list.append(clipped_surf_prox)

#     surf_prox = np.concatenate(surf_prox_list)
#     return torch.from_numpy(surf_prox)

def _coord(pdb_line):
    x=float(pdb_line[30:38].strip())
    y=float(pdb_line[38:46].strip())
    z=float(pdb_line[46:54].strip())
    return x,y,z

def compute_surface_proximities(pdb_dict,chain_atom_idt,interface_atom_idt):
    parser=PDBParser()
    structure=parser.get_structure('complex',io.StringIO(pdb_dict['str_pdb']))
    model=structure[0]
    surfaces={chain.id:get_surface(chain) for chain in model}
    
    atom_depth_map=dict()
    for chain,atoms_index in chain_atom_idt.items():
        for atom_index in atoms_index:
            if atom_index in interface_atom_idt['res']:
                coord=_coord(pdb_dict['dict_pdb'][atom_index])
                atom_depth_map[atom_index]=min_dist(np.array(coord),surfaces[chain])
    
    surf_prox_list=[atom_depth_map[atom_index] for atom_index in interface_atom_idt['res']]
    surf_prox_list=np.array(surf_prox_list).reshape(-1,1)
    surf_prox_list=np.ones_like(surf_prox_list)-max_normalize_array(surf_prox_list,max_value=100.)
    surf_prox_list=surf_prox_list.clip(0.,surf_prox_list.max())
    return torch.tensor(surf_prox_list,dtype=torch.float32)
    

def assign_covalent_radii_to_atom_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign covalent radius to each atom based on its bond state using values from:
    Atomic Structures of all the Twenty Essential Amino Acids and a Tripeptide, with Bond Lengths as Sums of Atomic Covalent Radii
    Heyrovska, 2008

    :param df: Pandas PDB dataframe with a bond_states_column
    :type df: pd.DataFrame
    :return: Pandas PDB dataframe with added covalent_radius column
    :rtype: pd.DataFrame
    """
    # Assign covalent radius to each atom
    df["covalent_radius"] = df["atom_bond_state"].map(COVALENT_RADII)
    return df

def one_of_k_encoding_unk(feat, allowable_set):
    """Convert input to 1-hot encoding given a set of (or sets of) allowable values.
      Additionally, map inputs not in the allowable set to the last element."""
    if feat not in allowable_set:
        print('{} not in STD_ATOM_NAMES'.format(feat))
        feat = allowable_set[-1]
    return list(map(lambda s: feat == s, allowable_set))

def prot_df_to_dgl_graph_with_feats(df: pd.DataFrame, feats: List, allowable_feats: List[List], knn: int) \
        -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert protein in DataFrame representation to a graph compatible with DGL, where each node is an atom."""
    # Aggregate one-hot encodings of each atom's type to serve as the primary source of atom features
    atom_type_feat_vecs = [one_of_k_encoding_unk(feat, allowable_feats[0]) for feat in feats]
    atom_types = torch.FloatTensor(atom_type_feat_vecs)
    assert not torch.isnan(atom_types).any(), 'Atom types must be valid float values, not NaN'

    # Gather chain IDs to serve as an additional node feature
    chain_ids = torch.FloatTensor([c for c, (k, g) in enumerate(groupby(df['chain_id'].values.tolist()), 0) for _ in g])

    # Organize atom coordinates into a FloatTensor
    node_coords = torch.tensor(df[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)

    # Define edges - KNN argument determines whether an atom-atom edge gets created in the resulting graph
    knn_graph = dgl.knn_graph(node_coords, knn)

    return knn_graph, node_coords, atom_types, chain_ids

def determine_is_ca_atom(graph: dgl.DGLGraph, atom_df: pd.DataFrame) -> torch.Tensor:
    """
    Determine, for each atom, whether it is a carbon-alpha (Ca) atom or not.

    Parameters
    ----------
    graph: dgl.DGLGraph
    atom_df: pd.DataFrame

    Returns
    -------
    torch.Tensor
    """
    is_ca_atom = torch.zeros((graph.num_nodes(), 1), dtype=torch.bool)
    is_ca_atom_df_indices = atom_df[atom_df['atom_name'] == 'CA'].index.values
    is_ca_atom[is_ca_atom_df_indices] = True
    return is_ca_atom.int()

def derive_dihedral_angles(pdb_filepath: str) -> pd.DataFrame:
    """Find all dihedral angles for the residues in an input PDB file."""
    # Increase BioPython's MaxPeptideBond to capture all dihedral angles
    internal_coords.IC_Chain.MaxPeptideBond = 100.0

    # Parse our input PDB structure
    parser = PDBParser(PERMISSIVE=1, QUIET=1)
    structure = parser.get_structure('', pdb_filepath)

    # Generate internal coordinates for the input structure
    structure.atom_to_internal_coordinates()

    # Collect backbone dihedral angles for each residue
    num_residues = int(sum([len(record.seq) for record in SeqIO.parse(pdb_filepath, "pdb-atom")]))
    dihedral_angles_dict = {i + 1: np.zeros(3) for i in range(num_residues)}

    # Note: Each structure has at least two chains
    residue_num = 1
    latest_residue_num = 0
    structure_chains = list(structure.get_chains())
    for structure_chain in structure_chains:
        ic_chain = structure_chain.internal_coord
        for key in ic_chain.dihedra.keys():
            dihedral = ic_chain.dihedra[key]

            dihedral_id_tokens = dihedral.id.split(':')
            residue_num = int(dihedral_id_tokens[1].split('_')[0]) + latest_residue_num
            dihedral_angle_atoms = [s.split('_')[-1] for s in dihedral_id_tokens]

            if dihedral_angle_atoms in DIHEDRAL_ANGLE_ID_TO_ATOM_NAMES.values():
                angle_idx = [k for k, v in DIHEDRAL_ANGLE_ID_TO_ATOM_NAMES.items() if v == dihedral_angle_atoms][0]
                dihedral_angles_dict[residue_num][angle_idx] = ic_chain.dihedra[key].angle

        # Track the latest residue number in the most-recent chain
        latest_residue_num = residue_num

    # Assemble and return resulting dihedral angles
    dihedral_angles = np.stack(list(dihedral_angles_dict.values()), axis=0)
    assert dihedral_angles.any(), 'Must have found at least one valid dihedral angle for the input protein'
    return pd.DataFrame(dihedral_angles, columns=['phi', 'psi', 'omega'])

def compute_dihedral_angles(pdb_filepath: str) -> torch.Tensor:
    """
    Derive the phi and psi backbone dihedral angles for each residue in the input PDB file.

    Parameters
    ----------
    pdb_filepath: str

    Returns
    -------
    torch.Tensor
    """
    angles_df = derive_dihedral_angles(pdb_filepath)

    phi_angles = angles_df['phi'].values
    psi_angles = angles_df['psi'].values
    omega_angles = angles_df['omega'].values

    cos_encoded_angles = np.cos(phi_angles), np.cos(psi_angles), np.cos(omega_angles)
    sin_encoded_angles = np.sin(phi_angles), np.sin(psi_angles), np.sin(omega_angles)
    dihedral_angles = torch.from_numpy(np.stack((*cos_encoded_angles, *sin_encoded_angles), axis=1))
    return dihedral_angles

def compute_euclidean_distance_matrix(points: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between every input point (i.e., row).

    Parameters
    ----------
    points: torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    euclidean_dists = torch.norm(points[:, None] - points, dim=2, p=2)
    return euclidean_dists

def compute_bond_edge_feats(graph: dgl.DGLGraph, first_iter=False) -> torch.Tensor:
    """
    Compute edge feature indicating whether a covalent bond exists between a pair of atoms.

    Parameters
    ----------
    graph: dgl.DGLGraph
    first_iter: bool

    Returns
    -------
    torch.Tensor
    """
    # Compute all atom-atom Euclidean distances as a single distance matrix
    coords_distance_matrix = compute_euclidean_distance_matrix(graph.ndata['x_pred'])

    # Create a covalent 'distance' matrix by adding the radius array with its transpose
    orig_covalent_radius_distance_matrix = torch.add(
        graph.ndata['covalent_radius'].reshape(-1, 1),
        graph.ndata['covalent_radius'].reshape(1, -1)
    )

    # Add the covalent bond distance tolerance to the original covalent radius distance matrix
    covalent_radius_distance_matrix = (orig_covalent_radius_distance_matrix + COVALENT_RADIUS_TOLERANCE)

    # Sanity-check values in both distance matrices, only when first computing covalent bonds
    if first_iter:
        assert not torch.isnan(coords_distance_matrix).any(), 'No NaNs are allowed as coordinate pair distances'
        assert not torch.isnan(covalent_radius_distance_matrix).any(), 'No NaNs are allowed as covalent distances'

    # Threshold distance matrix to entries where Euclidean distance D > 0.4 and D < (covalent radius + tolerance)
    coords_distance_matrix[coords_distance_matrix <= 0.4] = torch.nan
    coords_distance_matrix[coords_distance_matrix >= covalent_radius_distance_matrix] = torch.nan
    covalent_bond_matrix = torch.nan_to_num(coords_distance_matrix)
    covalent_bond_matrix[covalent_bond_matrix > 0] = 1

    # Derive relevant covalent bonds based on the binary covalent bond matrix computed previously
    graph_edges_with_eids = graph.edges(form='all')
    graph_edges = torch.cat(
        (graph_edges_with_eids[0].reshape(-1, 1),
          graph_edges_with_eids[1].reshape(-1, 1)),
        dim=1
    )
    covalent_bond_edge_indices = covalent_bond_matrix.nonzero()
    combined_edges = torch.cat((graph_edges, covalent_bond_edge_indices))
    unique_edges, edge_counts = combined_edges.unique(dim=0, return_counts=True)
    covalently_bonded_edges = unique_edges[edge_counts > 1]

    # Find edges in the original graph for which a covalent bond was discovered
    covalently_bonded_eids = find_intersection_indices_2D(graph_edges, covalently_bonded_edges, graph_edges.device)

    # Craft new bond features based on the covalent bonds discovered above
    covalent_bond_feats = torch.zeros((len(graph_edges), 1))
    covalent_bond_feats[covalently_bonded_eids] = 1

    return covalent_bond_feats

def process_pdb_into_graph(input_filepath: str,
                            atom_selection_type: str,
                            knn: int,
                            idt: float,
                            pre_computed_feat: dict) -> dgl.DGLGraph:
    r""" Transform a given set of predicted and true atom DataFrames into a corresponding DGL graph.

    Parameters
    ----------
    input_filepath: str
    atom_selection_type: str
    knn: int
    idt: float

    Returns
    -------
    :class:`typing.Tuple[:class:`dgl.DGLGraph`, :class:`np.ndarray`]`
        Index 1. Graph structure, feature tensors for each node and edge.
            ...     predicted_node_coords = graph.ndata['x_pred']
            ...     true_node_coords = graph.ndata['x_true']
        - ``ndata['atom_type']``: one-hot type of each node
        - ``ndata['x_pred']:`` predicted Cartesian coordinate tensors of the nodes
        - ``ndata['x_true']:`` true Cartesian coordinate tensors of the nodes
        - ``ndata['labeled']``: one-hot indication of whether a node has a ground-truth coordinates label available
        - ``ndata['interfacing']``: one-hot indication of whether node lies within 'idt' Angstrom of an inter-chain node
        - ``ndata['covalent_radius']``: scalar descriptor of the hypothesized covalent radius of a node
        - ``ndata['chain_id']``: integer descriptor of the unique ID of the chain to which a node belongs (e.g., 3)
        - ``ndata['surf_prox']``: scalar descriptor of how close to the surface of a molecular chain a node is
        - ``edata['rel_pos']``: vector descriptor of the relative position of each destination node to its source node
        - ``edata['r']``: scalar descriptor of the normalized distance of each destination node to its source node
        - ``edata['w']``: scalar descriptor of the hypothetical weight between each destination node and source node
        - ``edata['edge_dist']``: zero-to-one normalized Euclidean distance between pairs of atoms
        - ``edata['bond_type']``: one-hot description of whether a hypothetical covalent bond exists between a node pair
        - ``edata['in_same_chain']``: one-hot description of whether a node pair belongs to the same molecular chain
        Index 2. DataFrame indices corresponding to atoms shared by both the predicted structure and true structure.
    """
    """Build the input graph"""
    # Determine atom selection type requested
    ca_only = atom_selection_type == 'ca_atom'

    # Ascertain atoms in DataFrames
    # input_pdb = PandasPdb().read_pdb(input_filepath)
    input_pdb = PandasPdb().read_pdb_from_list(input_filepath)
    input_atom_df = pd.concat([input_pdb.df['ATOM'],input_pdb.df['HETATM']])
    input_atom_df = input_atom_df.sort_values(by='line_idx').reset_index(drop=True)
    orig_pred_atom_df = copy.deepcopy(input_atom_df)

    # Reindex atom and residue numbers
    if ca_only:
        input_atom_df = input_atom_df[input_atom_df['atom_name'] == 'CA'].reset_index(drop=True)
    input_atom_df = reindex_df_field_values(input_atom_df, field_name='atom_number', start_index=1)
    input_atom_df = reindex_df_field_values(input_atom_df, field_name='residue_number', start_index=1)
    input_atom_df = reindex_df_field_values(input_atom_df, field_name='chain_id', start_index=0)

    # Get interfacing atom indices
    # orig_pred_df_indices = get_shared_df_coords(input_atom_df, input_atom_df, merge_order='right')
    # interfacing_atom_indices, _ = get_interfacing_atom_indices(input_atom_df, orig_pred_df_indices, idt)

    # if not ca_only:
    #     # Assign bond states to the dataframe, and then map these bond states to covalent radii for each atom
    #     # Note: For example, in the case of ligand input, set the 'residue_name' value for all atoms to 'XXX'
    #     input_atom_df = assign_bond_states_to_atom_dataframe(input_atom_df)
    #     input_atom_df = assign_covalent_radii_to_atom_dataframe(input_atom_df)

    # Construct KNN graph
    feats = input_atom_df['residue_name' if ca_only else 'atom_name']
    feats = feats.map(RESI_THREE_TO_1).tolist() if ca_only else feats
    try:
        graph, node_coords, atoms_types, chain_ids = prot_df_to_dgl_graph_with_feats(
            input_atom_df,  # All predicted atoms when constructing the initial graph
            feats,  # Which atom selection type to use for node featurization
            get_allowable_feats(ca_only),  # Which feature values are expected
            knn + 1  # Since we do not allow self-loops, we must include extra nearest neighbor to be removed thereafter
        )
    except DGLError:
        raise DGLError(f'In process_pdb_into_graph(), found an empty point set for {input_filepath}')

    # Remove self-loops in graph
    graph = dgl.remove_self_loop(graph)  # By removing self-loops w/ k=21, we are effectively left w/ edges for k=20

    # Manually add isolated nodes (i.e. those with no connected edges) to the graph
    if len(atoms_types) > graph.number_of_nodes():
        num_of_isolated_nodes = len(atoms_types) - graph.number_of_nodes()
        raise ValueError(f'{num_of_isolated_nodes} isolated node(s) detected in {input_filepath}')

    """Encode node features and labels in graph"""
    # Include one-hot features indicating each atom's type
    graph.ndata['atom_type'] = atoms_types  # [num_nodes, num_node_feats=21 if ca_only is True, 38 otherwise]
    # Cartesian coordinates for each atom
    graph.ndata['x_pred'] = node_coords  # [num_nodes, 3]
    # # One-hot ID representation of the atoms for which true coordinates were available in the corresponding exp. struct.
    # graph.ndata['labeled'] = torch.zeros((graph.number_of_nodes(), 1), dtype=torch.int32)
    # # One-hot ID representation of the atoms present in any interface between at least two chains
    # interfacing = torch.zeros((graph.number_of_nodes(), 1), dtype=torch.int32)
    # interfacing[interfacing_atom_indices] = 1
    # graph.ndata['interfacing'] = interfacing  # [num_nodes, 1]
    # # Single scalar describing the covalent radius of each atom
    # null_cov_radius = torch.zeros_like(graph.ndata['interfacing'])
    # covalent_radius = null_cov_radius if ca_only else torch.FloatTensor(input_atom_df['covalent_radius']).reshape(-1, 1)
    # graph.ndata['covalent_radius'] = covalent_radius
    # Single value indicating to which chain an atom belongs
    graph.ndata['chain_id'] = chain_ids.reshape(-1, 1)  # [num_nodes, 1]
    # Integers describing to which residue an atom belongs
    graph.ndata['residue_number'] = torch.IntTensor(input_atom_df['residue_number'].tolist())  # [num_nodes, 1]
    # Scalars describing each atom's proximity to the surface of its chain
    # graph.ndata['surf_prox'] = compute_surface_proximities(graph, input_filepath, ca_only)  # [num_nodes, 1]
    graph.ndata['surf_prox'] = pre_computed_feat['proximity']
    # graph.ndata['dpx_cx_lap'] = pre_computed_feat['dpx_cx_lap']
    # # Binary integers (i.e., 0 or 1) describing whether an atom is a carbon-alpha (Ca) atom
    # graph.ndata['is_ca_atom'] = determine_is_ca_atom(graph, input_atom_df.reset_index(drop=True))  # [num_nodes, 1]
    # # Scalars describing the cosine and sine-activated phi, psi, and omega backbone dihedral angles for each residue
    # dihedral_angles = compute_dihedral_angles(input_filepath) if ca_only else torch.zeros_like(graph.ndata['x_pred'])
    # graph.ndata['dihedral_angles'] = dihedral_angles  # [num_nodes, 6]
    # add atom mass and charge
    graph.ndata['mass_charge']=mass_and_charge(io.StringIO(''.join(input_filepath)))#(input_atom_df[['residue_name','atom_name']])
    graph.ndata['moltype']=pre_computed_feat['moltype']
    
    graph.ndata['atom_number_ori']=torch.tensor(orig_pred_atom_df['atom_number'].values)
    graph.ndata['residue']=torch.IntTensor([ENCODE_RES.index(x) if x in ENCODE_RES else -1 for x in input_atom_df['residue_name'].tolist()])

    """Encode edge features in graph"""
    # Positional encoding for each edge (used for sequentially-ordered inputs like proteins)
    graph.edata['pos_enc'] = torch.sin((graph.edges()[0] - graph.edges()[1]).float()).reshape(-1, 1)  # [num_edges, 1]
    # update_relative_positions(graph)  # Relative positions of nodes j to nodes i - [num_edges, 1]
    # update_potential_values(graph, r=graph.edata['r'])  # Weights from nodes j and nodes i - [num_edges, 1]
    # bond_types = torch.zeros_like(graph.edata['r']) if ca_only else compute_bond_edge_feats(graph, first_iter=True)
    # graph.edata['bond_type'] = bond_types.reshape(-1, 1)  # [num_edges, 1]
    graph.apply_edges(compute_chain_matches)  # Install edata['in_same_chain'] - [num_edges, 1]
    rel_geom_feats = compute_rel_geom_feats(graph, input_atom_df, input_file=input_filepath)
    graph.edata['rel_geom_feats'] = rel_geom_feats  # [num_edges, 12]
    graph.apply_edges(compute_molecular_matches)

    """Ensure no input feature values are invalid (e.g., NaN)"""
    nans_in_ndata = torch.isnan(graph.ndata['x_pred']).any() \
                    or torch.isnan(graph.ndata['atom_type']).any() \
                    or torch.isnan(graph.ndata['chain_id']).any() \
                    or torch.isnan(graph.ndata['surf_prox']).any() \
                    or torch.isnan(graph.ndata['mass_charge']).any() \
                    or torch.isnan(graph.ndata['moltype']).any()
                    # or torch.isnan(graph.ndata['dpx_cx_lap']).any() \
                    
                    # or torch.isnan(graph.ndata['labeled']).any() \
                    # or torch.isnan(graph.ndata['interfacing']).any() \
                    # or torch.isnan(graph.ndata['covalent_radius']).any() \
                    # or torch.isnan(graph.ndata['residue_number']).any() \
                    
                    # or torch.isnan(graph.ndata['is_ca_atom']).any() \
                    # or torch.isnan(graph.ndata['dihedral_angles']).any()
    nans_in_edata = torch.isnan(graph.edata['pos_enc']).any() \
                    or torch.isnan(graph.edata['in_same_chain']).any() \
                    or torch.isnan(graph.edata['rel_geom_feats']).any() \
                    or torch.isnan(graph.edata['in_same_molecular']).any()
                    # or torch.isnan(graph.edata['rel_pos']).any() \
                    # or torch.isnan(graph.edata['r']).any() \
                    # or torch.isnan(graph.edata['bond_type']).any() \
    assert not (nans_in_ndata or nans_in_edata), 'There must be no invalid (i.e., NaN) values in the graph features'

    # Return our resulting graph
    return graph

def atom_index_pdb(pdb):
    pdb=list(filter(lambda x:re.match('ATOM',x) or re.match('HETATM',x),pdb.split('\n')))
    atom_index_line={int(l[6:11].strip()):l for l in pdb}
    return atom_index_line

def fixedpdb_chain_interface_index(pdbfile,interact_dist):
    if not os.path.exists(pdbfile.replace('.pdb','_fix.pdb')):
        os.system('pdbfixer {} --output={} --add-atoms=heavy'.format(
            pdbfile,pdbfile.replace('.pdb','_fix.pdb')))
    pymol.cmd.delete('all')
    pymol.cmd.load(pdbfile.replace('.pdb','_fix.pdb'))
    reindexed_pdb=pymol.cmd.get_pdbstr() #reindex atom, keep residue
    pymol.cmd.delete('all')
    pymol.cmd.read_pdbstr(reindexed_pdb,'reindexed_pdb')
    chains=pymol.cmd.get_chains()
    chain_atom_idt={chain:pymol.cmd.identify('chain {}'.format(chain)) for chain in chains}
    pymol.cmd.select('proatoms','polymer.protein within {} of polymer.nucleic'.format(interact_dist))
    pymol.cmd.select('rnaatoms','polymer.nucleic within {} of polymer.protein'.format(interact_dist))
    interface_atom_idt={'atom':pymol.cmd.identify('proatoms')+pymol.cmd.identify('rnaatoms'),
                        'res':pymol.cmd.identify('byres proatoms')+pymol.cmd.identify('byres rnaatoms')}
    molecular_atom_idt={'protein':pymol.cmd.identify('polymer.protein'),'rna':pymol.cmd.identify('polymer.nucleic')}
    pymol.cmd.delete('all')
    return {'str_pdb':reindexed_pdb,'dict_pdb':atom_index_pdb(reindexed_pdb)},chain_atom_idt,interface_atom_idt,molecular_atom_idt

def run_naccess(
    model, pdb_file, probe_size=None, z_slice=None, hetatm=None, naccess="naccess", temp_path="/tmp/"
):
    """Run naccess for a pdb file."""
    # make temp directory;
    tmp_path = tempfile.mkdtemp(dir=temp_path)

    # file name must end with '.pdb' to work with NACCESS
    # -> create temp file of existing pdb
    #    or write model to temp file
    handle, tmp_pdb_file = tempfile.mkstemp(".pdb", dir=tmp_path)
    os.close(handle)
    if pdb_file:
        pdb_file = os.path.abspath(pdb_file)
        shutil.copy(pdb_file, tmp_pdb_file)
    else:
        writer = PDBIO()
        writer.set_structure(model.get_parent())
        writer.save(tmp_pdb_file)

    # chdir to temp directory, as NACCESS writes to current working directory
    old_dir = os.getcwd()
    os.chdir(tmp_path)

    # create the command line and run
    # catch standard out & err
    command = [naccess, tmp_pdb_file]
    if probe_size:
        command.extend(["-p", str(probe_size)])
    if z_slice:
        command.extend(["-z", z_slice])
    if hetatm:
        command.extend(["-h"])

    p = subprocess.Popen(
        command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = p.communicate()
    os.chdir(old_dir)

    rsa_file = tmp_pdb_file[:-4] + ".rsa"
    asa_file = tmp_pdb_file[:-4] + ".asa"
    # Alert user for errors
    if err.strip():
        warnings.warn(err)

    if (not os.path.exists(rsa_file)) or (not os.path.exists(asa_file)):
        raise Exception("NACCESS did not execute or finish properly.")

    # get the output, then delete the temp directory
    with open(rsa_file) as rf:
        rsa_data = rf.readlines()
    with open(asa_file) as af:
        asa_data = af.readlines()
    # shutil.copy(asa_file,'./asa.txt')
    shutil.rmtree(tmp_path, ignore_errors=True)
    return rsa_data, asa_data

def process_asa_data(asa_data):
    """Process the .asa output file: atomic level SASA data."""
    asa_values=[float(l[54:62].strip()) for l in asa_data]
    return np.array(asa_values)

def calculate_dpx(coordinates, asa_values):
    print(time.time())
    assert len(coordinates)==len(asa_values)
    asa_gt_zero_indices = np.where(asa_values > 0)[0]
    asa_eq_zero_indices = np.where(asa_values == 0)[0]
    
    dpx = np.zeros(len(asa_values))
    if len(asa_gt_zero_indices) > 0:
        dpx[asa_gt_zero_indices] = 0

    if len(asa_eq_zero_indices) > 0:
        asa_gt_zero_coords = coordinates[asa_gt_zero_indices]
        for i in asa_eq_zero_indices:
            distances = scidist.cdist(coordinates[i:i+1], asa_gt_zero_coords)
            dpx[i] = np.min(distances)
    print(time.time())
    return dpx

def calculate_cx(coordinates, radius=10):
    print(time.time())
    cx = []
    for center_coord in coordinates:
        atom_count = 0

        for coord in coordinates:
            if np.linalg.norm(center_coord - coord) <= radius:
                atom_count += 1

        volume_int = atom_count * 20.1
        volume_sphere = 4/3 * np.pi * radius**3

        Vext = volume_sphere - volume_int
        current_cx = Vext / volume_int
        cx.append(current_cx)
    print(time.time())
    return np.array(cx)

def calculate_dpx_latest(coordinates, asa_values):
    assert len(coordinates)==len(asa_values)
    asa_gt_zero_indices = np.where(asa_values > 0)[0]
    asa_eq_zero_indices = np.where(asa_values == 0)[0]
    
    dpx = np.zeros(len(asa_values))
    if len(asa_gt_zero_indices) > 0:
        dpx[asa_gt_zero_indices] = 0
    
    if len(asa_eq_zero_indices) > 0:
        asa_gt_zero_coords = coordinates[asa_gt_zero_indices]
        asa_eq_zero_coords = coordinates[asa_eq_zero_indices]
        distances=scidist.cdist(asa_eq_zero_coords,asa_gt_zero_coords,'euclidean')
        dpx[asa_eq_zero_indices]=np.min(distances,axis=1)
    return dpx

def calculate_cx_latest(coordinates, radius=10):
    dist_tree=spatial.cKDTree(coordinates)
    num_points=dist_tree.query_ball_point(coordinates,r=radius,return_length=True)
    volume_sphere = 4/3 * np.pi * radius**3
    cx=list(map(lambda x:volume_sphere/(x*20.1)-1,num_points))
    return np.array(cx)

def Laplacian(coordinates):
    def _lap(distmap,coord):
        distancelist=np.triu(distmap,k=1)
        distancelist=distancelist[np.nonzero(distancelist)].tolist()
        distancelist.sort()
        points1,points2=int(len(distancelist)/4),int(len(distancelist)/2)
        points3=points1+points2
        pointlist=[distancelist[i] for i in (0,points1,points2,points3,len(distancelist)-1)]
        
        omaplist=[]
        for point in pointlist:
            omap=np.exp(-np.square(distmap/point))
            omaplist.append(np.triu(omap,k=2)+np.tril(omap,k=-2))
        
        normlist=[]
        for omap in omaplist:
            olist=np.sum(omap,axis=1)
            res_normlist=list()
            for i in range(len(olist)):
                pi=coord[i,:]
                pj=np.sum(coord*omap[i,:].reshape(-1,1),axis=0)/olist[i]
                res_normlist.append(np.sqrt(np.sum(np.square(pi-pj))))
            res_normlist=np.array(res_normlist)
            normlist.append(res_normlist)
        normlist=np.column_stack(tuple(normlist))
        return normlist
    
    distmap=scidist.cdist(coordinates,coordinates,'euclidean')
    laps=_lap(distmap,coordinates)
    laps_normalized=(laps-np.mean(laps,axis=0))/np.std(laps,axis=0)
    return laps_normalized

def compute_atomic_dpx_cx_lap(pdb_dict,chain_atom_idt,interface_atom_idt):
    dpx_cx=dict()
    lap=dict()
    for chain,atom_index in chain_atom_idt.items():
        monomer='\n'.join([pdb_dict['dict_pdb'][x] for x in atom_index])
        # with open('./amonomer.txt','w') as f:
        #     f.write(monomer)
        p=PDBParser()
        structure=p.get_structure('monomer',io.StringIO(monomer))
        model=structure[0]
        _,asa_data=run_naccess(model=model,pdb_file=None,probe_size=1.5,hetatm=True)
        coordinates=np.array([_coord(pdb_dict['dict_pdb'][x]) for x in atom_index],dtype=float)
        dpx=calculate_dpx_latest(coordinates, process_asa_data(asa_data))
        cx=calculate_cx_latest(coordinates)
        dpx_cx.update(dict(zip(atom_index,np.column_stack((dpx,cx)).tolist())))
        chain_lap=Laplacian(coordinates)
        lap.update(dict(zip(atom_index,chain_lap.tolist())))
    
    atomic_dpx_cx=np.array([dpx_cx[x] for x in interface_atom_idt['res']],dtype=np.float64)
    atomic_dpx_cx=np.ones_like(atomic_dpx_cx)-max_normalize_array(atomic_dpx_cx,max_value=5.)
    atomic_dpx_cx=atomic_dpx_cx.clip(0,atomic_dpx_cx.max())
    atomic_lap=np.array([lap[x] for x in interface_atom_idt['res']],dtype=np.float64)
    atomic_lap=atomic_lap.clip(-5.,5.)
    atomic_dpx_cx_lap=np.column_stack((atomic_dpx_cx,atomic_lap))
    
    return torch.tensor(atomic_dpx_cx_lap,dtype=torch.float32)

def get_pdb_moltype(pdb_dict,interface_atom_idt,molecular_atom_idt):
    pdb=list()
    moltype=list()
    for idt in interface_atom_idt['res']:
        pdb.append(pdb_dict['dict_pdb'][idt]+'\n')
        if idt in molecular_atom_idt['protein']:
            moltype.append(0)
        if idt in molecular_atom_idt['rna']:
            moltype.append(1)
    assert len(pdb)==len(moltype)
    return pdb,np.array(moltype,dtype=int)

def get_atom_chain_resindex(pdb_dict,chain_atom_idt):
    atom_chain_resindex=dict()#key->atom num value->list[chain, residue num]
    for chain,atom_index in chain_atom_idt.items():
        monomer=[pdb_dict['dict_pdb'][x] for x in atom_index]
        monomer = PandasPdb().read_pdb_from_list(monomer)
        chain_atom_df = pd.concat([monomer.df['ATOM'],monomer.df['HETATM']])
        chain_atom_df = chain_atom_df.sort_values(by='line_idx').reset_index(drop=True)
        
        chain_atom_df=reindex_df_field_values(chain_atom_df, field_name='residue_number', start_index=0)
        atom_chain_resindex.update(dict(zip(chain_atom_df['atom_number'].values.astype(int).tolist(),
                                      chain_atom_df[['chain_id','residue_number']].values.astype(str).tolist())))
    return atom_chain_resindex

    