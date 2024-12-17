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


