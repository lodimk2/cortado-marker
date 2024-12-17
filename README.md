# CORTADO: hill Climbing Optimization foR cell-Type specific mArker gene DiscOvery


![github_flowchart](https://github.com/user-attachments/assets/8fec5bc5-fd99-47cb-a566-cbf1c69e1370)


## Description 

ORTADO is a marker gene selection framework with three main steps. First, a single cell genomics count matrix is loaded as a Scanpy object, and undergoes standard preprocessing. Then, the stochastic hill optimization is run, with four key components taken into consideration when selecting genes: Differential expressed gene score, non redundancy based on cosine similarity, a penalization of selecting too many genes, and a constraint parameter to select a user defined amount of genes. Then, the genes can be visualized through expression heatmaps, and contextualized using relevant literature and gene set enrichment analyses. 
