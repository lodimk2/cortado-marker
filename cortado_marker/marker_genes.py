import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import numpy as np 

def calc_marker_gene_score(adata, target_cluster, n_genes, pval_threshold=0.05, use_raw=False):
    """
    Calculate marker gene scores for a given cluster.

    Parameters:
    - adata (AnnData): Gene expression data
    - target_cluster (str): Target cluster label
    - n_genes (int): Number of top marker genes to select
    - pval_threshold (float): P-value threshold for marker genes
    - use_raw (bool): If True, use raw data for marker gene selection

    Returns:
    - marker_scores (pd.DataFrame): Marker gene scores
    """
    sc.tl.rank_genes_groups(
        adata,
        groupby='clust_assign',
        method='wilcoxon',
        reference='rest',
        n_genes=n_genes,
        use_raw=use_raw,
        tie_correct=True,
        corr_method='bonferroni'
    )
    
    # Step 2: Extract log2 fold changes for the target cluster
    target_log2fc = pd.DataFrame(adata.uns['rank_genes_groups']['logfoldchanges'])[target_cluster]
    genes = pd.DataFrame(adata.uns['rank_genes_groups']['names'])[target_cluster]
    # Step 3: Calculate the mean log2 fold change for each gene across all other clusters
    log2fc_all_clusters = pd.DataFrame(adata.uns['rank_genes_groups']['logfoldchanges'])
    mean_log2fc_others = log2fc_all_clusters.drop(columns=[target_cluster]).mean(axis=1)
    
    # Step 4: Calculate the marker score
    scores_df = pd.DataFrame({
        'gene': genes,
        'log2fc_target': target_log2fc,
        'pval': adata.uns['rank_genes_groups']['pvals'][target_cluster],
        'pval_adj': adata.uns['rank_genes_groups']['pvals_adj'][target_cluster],
        'mean_log2fc_others': mean_log2fc_others,
        'marker_score': target_log2fc - mean_log2fc_others
    })
    filtered_df = scores_df[scores_df['pval_adj'] < pval_threshold].dropna()
    
    # Step 5: Sort the DataFrame by marker score in descending order
    scores_df = filtered_df.sort_values(by='marker_score', ascending=False).set_index('gene')
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    scaled_scores_df = scores_df
    # Fit and transform the marker_score column
    scaled_scores_df['marker_score'] = scaler.fit_transform(scores_df[['marker_score']])

    return scaled_scores_df[['marker_score']]

def gene_correlation_within_cluster(target_cluster, adata):
    """
    Calculate gene correlation within a given cluster.

    Parameters:
    - target_cluster (str): Target cluster label
    - adata (AnnData): Gene expression data

    Returns:
    - sim_scores (pd.DataFrame): Gene correlation scores
    """
    cells_in_cluster = np.where(adata.obs['clust_assign'] == target_cluster)[0]
    #print("Cells in cluster", cells_in_cluster)
    #print("Printing norm exp")
    # Subset the sparse normalized expression data to cells in the cluster of interest
    clust_exp = adata.X[cells_in_cluster, :].T  # Transpose to have genes as rows
    
    # Ensure clust_exp is a dense matrix for cosine_similarity calculation
    if isinstance(clust_exp, csr_matrix):
        clust_exp = clust_exp.toarray()
    #print(clust_exp)
    # Compute cosine similarity matrix between genes
    similarity_matrix = cosine_similarity(clust_exp)
    
    # Convert the similarity matrix to a pandas DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=adata.var_names, columns=adata.var_names)
    
    return similarity_df
