# CORTADO: hill Climbing Optimization foR cell-Type specific mArker gene DiscOvery


![github_flowchart](https://github.com/user-attachments/assets/8fec5bc5-fd99-47cb-a566-cbf1c69e1370)


## Description 

CORTADO performs marker gene selection through three main steps:

1. **Loading and Preprocessing:**
   - A single-cell genomics count matrix is loaded as a **Scanpy object**.
   - Standard preprocessing steps are applied.

2. **Stochastic Hill Optimization:**
   Gene selection is optimized considering four key components:
   - **Differential Expression Score**: Prioritizes genes with significant differential expression.
   - **Non-Redundancy**: Avoids redundancy using cosine similarity.
   - **Penalization for Gene Count**: Penalizes selecting too many genes.
   - **Constraint Parameter**: Allows selection of a user-defined number of genes.

3. **Visualization and Contextualization:**
   - Selected genes can be visualized using **expression heatmaps**.
   - Genes are contextualized through:
     - Relevant literature.
     - Gene Set Enrichment Analyses (GSEA).


## Installation

CORTADO can be installed using the PyPi installer (pip)

``` pip install cortado-marker ```. 

To import CORTADO in your Python code, use the following import statement:

``` import cortado-marker as cortado ```

## Tutorial 

### Load Data

CORTADO has a custom load function, ``` load_data() ```. The load data requires some specific parameters:

```python
adata = cortado.load_data(exp_path, metadata_path, metadata_label_column, tenX=False, preprocess=True) 

#### Required Parameters:
- **`exp_path`** *(str)*: Path to the gene expression matrix file (e.g., counts matrix in CSV format).  
- **`metadata_path`** *(str)*: Path to the metadata file (e.g., cell annotations in CSV format).  
- **`metadata_label_column`** *(str)*: The column in the metadata file containing group labels.  

#### Optional Parameters:
- **`tenX`** *(bool, default=False)*: Set to `True` if the input data is in 10X format.  
- **`preprocess`** *(bool, default=True)*: Whether to preprocess the data (log transformation and scaling).  
```
### Calculate Marker Gene Scores and Cosine Similarity 

Next, we calculate the marker gene scores and the cosine similarity vectors for each gene. There are important paramters for customization here as well. 

```python
# Parameters
adata = <AnnData object>  # Input single-cell dataset
target_cluster = "Cluster_1"  # The name or label of the target cluster
n_genes = 50  # Number of top marker genes to select
p_val_threshold = 0.05  # P-value threshold for filtering marker genes
use_raw = True  # Whether to use raw gene expression data

# Calculate marker gene scores for the current cluster
marker_scores = cortado.calc_marker_gene_score(adata, target_cluster, n_genes, p_val_threshold, use_raw=use_raw)
print(f"Calculated marker scores for cluster {target_cluster}")

# Calculate gene correlation within the current cluster
sim_scores = cortado.gene_correlation_within_cluster(target_cluster, adata)
print(f"Calculated gene correlation for cluster {target_cluster}")

# Filter correlation matrix for the marker genes
filtered_genes = marker_scores.index
filtered_corr_matrix = sim_scores.loc[filtered_genes, filtered_genes]
filtered_corr_matrix = filtered_corr_matrix.reindex(index=marker_scores.index, columns=marker_scores.index)

```

### Obtain Marker Genes 

Finally, we obtain the marker genes for a given cluster using the ```python cortado.run_stochastic_hill_climbing()``` function. 

```python
# Parameters
how_many = 20  # Number of marker genes to select
max_iterations = 1000  # Maximum number of iterations for the optimization
gamma = 0.5  # Weight for the cosine similarity penalty
idle_limit = 50  # Idle limit for stopping if no improvement
lambda1 = 1.0  # Weight for the differential expression score
lambda2 = 1.0  # Weight for non-redundancy (cosine similarity penalty)
lambda3 = 0.5  # Penalty for selecting too many genes
mode = 0  0 for unconstrained, 1 for constrained

# Run stochastic hill climbing for the current cluster
best_solution, best_value = cortado.run_stochastic_hill_climbing(
    marker_scores, filtered_corr_matrix, how_many=how_many, max_iterations=max_iterations, 
    gamma=gamma, idle_limit=idle_limit, lambda1=lambda1, lambda2=lambda2, 
    lambda3=lambda3, mode=mode, plot_filename=""
)

# Output the results
print(f"Best Solution: {best_solution}")
print(f"Best Value: {best_value}")

# Extract gene names where best_solution equals 1
selected_genes = marker_scores.index[best_solution == 1].tolist()
        
# Store the selected genes for this cluster
all_selected_genes[target_cluster] = selected_genes
```


## Citation 

If you have used CORTADO, please consider citing our manuscript. 

## Support 

For any questions, comments or concerns, please reach out to Musaddiq Lodi @ lodimk2@vcu.edu
