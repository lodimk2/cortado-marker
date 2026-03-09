import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import numpy as np 

def calc_marker_gene_score(adata, target_cluster, n_genes, p_val_threshold, cluster_column, use_raw=True, n_hvg=2000):
    """
    Calculate marker gene scores for a target cluster.
    
    Parameters:
    - adata: AnnData object
    - target_cluster: Target cluster identifier
    - n_genes: Number of top genes to return
    - p_val_threshold: P-value threshold for filtering
    - use_raw: Whether to use raw data for calculations
    - n_hvg: Number of highly variable genes to consider
    """
    # Work with a copy to avoid modifying original
    adata_work = adata.copy()
    
    # If adata has more genes than n_hvg, filter to HVG first
    if adata_work.n_vars > n_hvg and 'highly_variable' not in adata_work.var.columns:
        # Calculate HVG if not already done
        if use_raw and adata_work.raw is not None:
            # Use raw data for HVG calculation
            adata_temp = adata_work.raw.to_adata()
            sc.pp.normalize_total(adata_temp, target_sum=1e4)
            sc.pp.log1p(adata_temp)
            sc.pp.highly_variable_genes(adata_temp, n_top_genes=n_hvg)
            hvg_genes = adata_temp.var_names[adata_temp.var.highly_variable].tolist()
        else:
            # Calculate HVG on current data
            sc.pp.highly_variable_genes(adata_work, n_top_genes=n_hvg)
            hvg_genes = adata_work.var_names[adata_work.var.highly_variable].tolist()
        
        # Filter to HVG
        adata_work = adata_work[:, hvg_genes]
    
    # Now run differential expression on filtered data
    sc.tl.rank_genes_groups(
        adata_work, 
        cluster_column, 
        groups=[str(target_cluster)],
        reference='rest',
        method='wilcoxon',
        use_raw=use_raw
    )
    
    # Extract results
    result = adata_work.uns['rank_genes_groups']
    genes = result['names'][str(target_cluster)][:n_genes]
    scores = result['scores'][str(target_cluster)][:n_genes]
    pvals = result['pvals_adj'][str(target_cluster)][:n_genes]
    
    # Filter by p-value
    mask = pvals < p_val_threshold
    genes = genes[mask]
    scores = scores[mask]
    
    # Return as DataFrame
    marker_scores = pd.DataFrame({
        'gene': genes,
        'score': scores
    }).set_index('gene')['score']
    
    return marker_scores

def gene_correlation_within_cluster(target_cluster, adata, n_hvg=2000):
    """
    Calculate gene correlation within a specific cluster.
    
    Parameters:
    - target_cluster: Target cluster identifier
    - adata: AnnData object
    - n_hvg: Number of highly variable genes to consider
    """
    # Filter to target cluster
    cluster_mask = adata.obs['clust_assign'] == str(target_cluster)
    adata_cluster = adata[cluster_mask, :].copy()
    
    # If we have more genes than n_hvg, filter to HVG
    if adata_cluster.n_vars > n_hvg:
        if 'highly_variable' not in adata_cluster.var.columns:
            # Calculate HVG for this subset
            sc.pp.highly_variable_genes(adata_cluster, n_top_genes=n_hvg)
        
        # Filter to HVG
        hvg_mask = adata_cluster.var.highly_variable
        adata_cluster = adata_cluster[:, hvg_mask]
    
    # Calculate correlation matrix
    if hasattr(adata_cluster.X, 'toarray'):
        expression_matrix = adata_cluster.X.toarray().T  # Genes x Cells
    else:
        expression_matrix = adata_cluster.X.T
    
    # Calculate correlation
    correlation_matrix = pd.DataFrame(
        np.corrcoef(expression_matrix),
        index=adata_cluster.var_names,
        columns=adata_cluster.var_names
    )
    
    return correlation_matrix