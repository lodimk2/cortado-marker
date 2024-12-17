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
load_data(exp_path, metadata_path, metadata_label_column, tenX=False, preprocess=True) 

#### Required Parameters:
- **`exp_path`** *(str)*: Path to the gene expression matrix file (e.g., counts matrix in CSV format).  
- **`metadata_path`** *(str)*: Path to the metadata file (e.g., cell annotations in CSV format).  
- **`metadata_label_column`** *(str)*: The column in the metadata file containing group labels.  

#### Optional Parameters:
- **`tenX`** *(bool, default=False)*: Set to `True` if the input data is in 10X format.  
- **`preprocess`** *(bool, default=True)*: Whether to preprocess the data (log transformation and scaling).  
```



