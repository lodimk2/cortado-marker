import argparse
import pandas as pd
import scipy as sc
import numpy as np
import random
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import os

def load_data(exp_path, metadata_path, metadata_label_column, tenX=False, preprocess=False):
    """
    Load gene expression data and metadata.

    Parameters:
    - exp_path (str): Path to gene expression CSV file
    - metadata_path (str): Path to metadata CSV file
    - metadata_label_column (str): Column in metadata that contains cell labels
    - tenX (bool): If True, indicates the data is in 10X format
    - preprocess (bool): If True, preprocess the gene expression data

    Returns:
    - adata (AnnData): Loaded gene expression data
    """
    if tenX:
        adata = sc.read_10x_mtx(exp_path, var_names='gene_symbols', gex_only=True, cache=False)
    else:
        gene_expression_df = pd.read_csv(exp_path, index_col=0)
        # Ensure all data are numeric
        gene_expression_df = gene_expression_df.apply(pd.to_numeric)
        # Replace zero and negative counts with a small positive value to avoid issues in log transformation
        gene_expression_df = gene_expression_df.clip(lower=1e-6)
        adata = sc.AnnData(gene_expression_df)
    #adata.raw = adata 
    metadata_df = pd.read_csv(metadata_path)
    # Convert to AnnData
    
    adata.obs['clust_assign'] = metadata_df[metadata_label_column].values
    adata.obs['clust_assign'] = adata.obs['clust_assign'].astype(str)
    adata.obs['clust_assign'] = adata.obs['clust_assign'].astype('category')
    if preprocess==False:
        return adata
        
    else: # Preprocess
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata)
        #sc.pp.scale(adata)
        #sc.tl.pca(adata)
        #sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        #sc.tl.louvain(adata, resolution=0.5)
    #print(adata.X)
    print("Created adata object with ", adata.shape[0], "cells and", adata.shape[1], "features.")
    return adata