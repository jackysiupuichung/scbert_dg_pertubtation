"""Embed a preprocessed dataset using the pretrained model.

TODO: add option to indicate missing genes with <mask> or <pad> tokens?

N.B. binning happens just by converting the datatype from float to long...
"""
# -*- coding: utf-8 -*-
import argparse
import gzip
import tqdm
import pickle as pkl
from functools import reduce
import numpy as np
import pandas as pd
import pickle as pkl
import torch
from performer_pytorch import PerformerLM
import scanpy as sc
from utils import *


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    CLASS = args.bin_num + 2
    SEQ_LEN = args.gene_num + 1

    model = PerformerLM(
        num_tokens = CLASS,
        dim = 200,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        local_attn_heads = 0,
        g2v_position_emb = True
    )

    data = sc.read_h5ad('./data/ctrl_norman_preprocessed.h5ad')

    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)

    batch_size = data.shape[0]
    model.eval()

    with open('./data/norman/all_perts.pkl', 'rb') as f:
        all_perts = pkl.load(f)

    with open('./data/norman/norman_mask_df.pkl', 'rb') as f:
        norman_cl_mask = pkl.load(f)

    
    with open('./data/norman/norman_mask_dg_df.pkl', 'rb') as f:
        co_pert_mask = pkl.load(f)

    genes = data.var_names.tolist()

    print("Genes")
    print(genes[:10])
    print(len(genes))

    print("Norman co_pert mask")
    print(co_pert_mask.columns)
    print(len(co_pert_mask.columns))
    print(co_pert_mask.head())
    print(co_pert_mask.shape)
    num_true_values = co_pert_mask.sum(axis=0)
    print("Number of true values in each column:")
    print(num_true_values)
    
    print("All perts")
    print(all_perts)

    # Count the number of samples where both genes in a pair are perturbed
    perturbation_counts = {}

    for pair in co_pert_mask.columns:
        gene1, gene2 = pair.split('+')
        
        # Check if both genes exist in the mask
        if gene1 in norman_cl_mask.columns and gene2 in norman_cl_mask.columns:
            # Perform logical AND to get samples where both genes are perturbed
            both_perturbed = norman_cl_mask[gene1] & norman_cl_mask[gene2]
            
            # Count the number of True values (i.e., cells where both genes are perturbed)
            count = both_perturbed.sum()
            perturbation_counts[pair] = count
            

    # Print the summary of perturbation counts
    print("Summary of perturbations:")
    for pair, count in perturbation_counts.items():
        print(f"{pair}: {count} samples with both genes perturbed.")

    with torch.no_grad():
        all_pert_embs = {}
        
        matching_co_pert_pairs = []

    # Iterate over the co-perturbations (pairs like 'CBL+UBASH3B')
    for co_pert in co_pert_mask.columns:  # Assuming co_pert_mask contains co-perturbations
        gene1, gene2 = co_pert.split('+')  # Split the co-perturbation into two genes

        # Check if either gene in the pair is present in the all_perts list
        if gene1 in all_perts or gene2 in all_perts:
            matching_co_pert_pairs.append(co_pert)  # Append the pair if either gene matches
        

        non_exp_gene = norman_cl_mask.columns[norman_cl_mask.sum() == 0]
        print("Non-expressed genes:", len(non_exp_gene), non_exp_gene)
        # Now proceed with processing the reversed pairs
        for pair in matching_co_pert_pairs:
            print(f"Processing matched pair: {pair}")
            gene1, gene2 = pair.split('+')
            
            if pair in non_exp_gene:
                pass
            else:
                cl_mask = co_pert_mask[pair].values  # Use the new co-perturbation mask
                slice_adata = data[cl_mask]
                
                gene_index1 = genes.index(gene1)
                gene_index2 = genes.index(gene2)

                batch_size = slice_adata.shape[0]
                embs = []
                for index in tqdm.tqdm(range(batch_size)):
                    full_seq = slice_adata.X[index].toarray()[0]
                    
                    # Zero out the perturbed genes (both genes in the pair)
                    full_seq[gene_index1] = 0
                    full_seq[gene_index2] = 0

                    full_seq[full_seq > (CLASS - 2)] = CLASS - 2
                    full_seq = torch.from_numpy(full_seq).long()
                    full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
                    full_seq = full_seq.unsqueeze(0)
                    
                    emb = model(full_seq, return_encodings=True)
                    embs.append(emb.squeeze(0).cpu().numpy())
                
                if args.average_embeddings:
                    all_pert_embs[pair] = np.array(embs).mean(1)
                    print(all_pert_embs[pair].shape)
                else:
                    raise NotImplementedError("Not implemented yet")
    
    # save the embeddings to gzipped pkl files
    with gzip.open('data/norman/scbert_perturbation_embeddings.pkl.gz', 'wb') as f:
        pkl.dump(all_pert_embs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
    parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
    parser.add_argument("--model_path", type=str, default='data/panglao_pretrain.pth')
    parser.add_argument("--average_embeddings", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")

    args = parser.parse_args()
    main(args)