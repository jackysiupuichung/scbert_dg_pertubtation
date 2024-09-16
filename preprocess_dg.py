import argparse
import re
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def main(args):
    panglao = sc.read_h5ad('./data/panglao_10000.h5ad')
    data = sc.read_h5ad(args.input_file)
    def get_ensembl(xref):
        match = re.search("Ensembl:.+\|", xref)
        if match is not None:
            return match.group(0).split(":")[1].split("|")[0]
        else:
            return None

    gene_df = pd.read_csv("data/Homo_sapiens.gene_info", sep="\t")
    gene_df["ensembl"] = gene_df["dbXrefs"].apply(get_ensembl)
    gene_mapping = {
        tup[0]: tup[1] for tup in gene_df[~gene_df["ensembl"].isnull()][["ensembl", "Symbol"]].values
    }
    new_var_names = []
    mapped_genes = 0
    for n in data.var_names:
        if n in gene_mapping:
            new_var_names.append(gene_mapping[n])
            mapped_genes +=1
        else:
            new_var_names.append(n)
    data.var_names = new_var_names
    print(f"Mapped {mapped_genes} genes of {len(new_var_names)} in data; (total in reference data {len(panglao.var_names)})")
    counts = sparse.lil_matrix((data.X.shape[0],panglao.X.shape[1]),dtype=np.float32)
    ref = panglao.var_names.tolist()
    obj = data.var.gene_symbols.tolist()  # correct for norman

    total_genes_found = 0
    missing_genes = []
    for i in range(len(ref)):
        if ref[i] in obj:
            loc = obj.index(ref[i])
            counts[:,i] = data.X[:,loc]
            print(counts[:,i])
            total_genes_found += 1
        else:
            missing_genes.append(ref[i])
            # pass
            print("Gene not found in data: ", ref[i], " at index: ", i)

    print("Total genes found: ", total_genes_found)
    print("Missing genes: ", len(missing_genes), missing_genes)
    with open("missing_genes.txt", "w") as f:
        f.write("\n".join(missing_genes))

    counts = counts.tocsr()
    new = ad.AnnData(X=counts)
    new.var_names = ref
    new.obs_names = data.obs_names
    new.obs = data.obs
    new.uns = panglao.uns

    sc.pp.filter_cells(new, min_genes=200)
    sc.pp.normalize_total(new, target_sum=1e4)
    sc.pp.log1p(new, base=2)
    new.write(args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()
    main(args)