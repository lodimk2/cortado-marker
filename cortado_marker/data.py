import argparse
import pandas as pd
import scanpy as sc
import numpy as np
import random
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import os
from anndata import AnnData
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

def extract_gene_embeddings(
    model,
    adata,
    batch_size: int = 64,
    num_workers: int = 8,
    device: str = "cuda",
    gene_list: list = None,
):
    """
    Extract gene embeddings from a scPrint model for a given AnnData object.
    Args:
        model (scPrint): A loaded and trained scPrint model.
        adata (AnnData): AnnData containing cell x gene matrix.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of workers for DataLoader.
        device (str): Device to run on ("cuda" or "cpu").
        gene_list (list): Optional list of genes to restrict to. If None, uses model.genes.
    Returns:
        embeddings (np.ndarray): A numpy array of shape (n_cells, n_genes, embedding_dim)
        genes_processed (list): The list of genes processed in order.
    """
    model.eval()
    model.to(device)
    model.pred_log_adata = False
    
    # Determine which genes to use
    if gene_list is None:
        gene_list = model.genes
    else:
        gene_list = [g for g in gene_list if g in model.genes]
        if len(gene_list) == 0:
            raise ValueError("No overlap between provided gene_list and model.genes")
    
    # Set up dataset and dataloader
    if "organism_ontology_term_id" not in adata.obs:
        adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"  # Default to human
    
    from scdataloader import Collator
    from scdataloader.data import SimpleAnnDataset
    
    adataset = SimpleAnnDataset(adata, obs_to_output=["organism_ontology_term_id"])
    col = Collator(
        organisms=model.organisms,
        valid_genes=model.genes,
        max_len=0,
        how="some",
        genelist=gene_list,
    )
    dataloader = DataLoader(
        adataset,
        collate_fn=col,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    
    all_embeddings = []
    
    # Use autocast to ensure half precision if required by the model
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
        for batch in dataloader:
            gene_pos, expression, depth = (
                batch["genes"].to(device),
                batch["x"].to(device),
                batch["depth"].to(device),
            )
            # Run encode_only to get transformer outputs
            output = model(
                gene_pos=gene_pos,
                expression=expression,
                req_depth=depth,
                get_gene_emb=True,
            )
            # Extract gene embeddings: shape (B, num_genes, d_model)
            gene_embeddings = output["gene_embedding"]
            all_embeddings.append(gene_embeddings.cpu().numpy())
            
            del output
            torch.cuda.empty_cache()
    
    # Concatenate all the embeddings for all cells
    embeddings = np.concatenate(all_embeddings, axis=0)  # shape: (n_cells, n_genes, d_model)
    
    return embeddings, gene_list

def load_data(exp_path, metadata_path, metadata_label_column, tenX=False, preprocess=False, 
              use_scprint_embeddings=False, scprint_model_path=None, 
              scprint_batch_size=64, scprint_device="cuda", n_hvg=2000):
    """
    Load gene expression data and metadata with optional scPRINT embedding generation.
    
    Parameters:
    - exp_path (str): Path to gene expression CSV file
    - metadata_path (str): Path to metadata CSV file
    - metadata_label_column (str): Column in metadata that contains cell labels
    - tenX (bool): If True, indicates the data is in 10X format
    - preprocess (bool): If True, preprocess the gene expression data
    - use_scprint_embeddings (bool): If True, generate scPRINT embeddings
    - scprint_model_path (str): Path to scPRINT model checkpoint (if None, uses pretrained)
    - scprint_batch_size (int): Batch size for scPRINT embedding generation
    - scprint_device (str): Device for scPRINT ("cuda" or "cpu")
    - n_hvg (int): Number of top HVGs to use for ALL analysis (default: 2000)
    
    Returns:
    - adata (AnnData): Loaded gene expression data (filtered to HVGs, with embeddings if requested)
    """
    
    print("Loading data...")
    
    # Original CORTADO data loading logic
    if tenX:
        adata = sc.read_10x_mtx(exp_path, var_names='gene_symbols', gex_only=True, cache=False)
    else:
        gene_expression_df = pd.read_csv(exp_path, index_col=0)
        # Ensure all data are numeric
        gene_expression_df = gene_expression_df.apply(pd.to_numeric)
        # Replace zero and negative counts with a small positive value
        gene_expression_df = gene_expression_df.clip(lower=1e-6)
        adata = sc.AnnData(gene_expression_df)
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    adata.obs['clust_assign'] = metadata_df[metadata_label_column].values
    adata.obs['clust_assign'] = adata.obs['clust_assign'].astype(str)
    adata.obs['clust_assign'] = adata.obs['clust_assign'].astype('category')
    
    print(f"Loaded data: {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # Store original expression before any modifications
    adata.layers['original_counts'] = adata.X.copy()
    
    # Standard preprocessing if requested
    if preprocess:
        print("Preprocessing data...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
        print(f"After preprocessing: {adata.shape[0]} cells, {adata.var.highly_variable.sum()} HVG")
        
        # ALWAYS filter to HVGs for downstream CORTADO analysis
        print(f"Filtering to top {n_hvg} highly variable genes for CORTADO analysis...")
        hvg_mask = adata.var.highly_variable
        adata = adata[:, hvg_mask].copy()
        print(f"Filtered data: {adata.shape[0]} cells, {adata.shape[1]} HVG genes")
    else:
        # Even without preprocessing, we should select HVGs for efficiency
        print(f"Selecting top {n_hvg} highly variable genes (no normalization)...")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
        hvg_mask = adata.var.highly_variable
        n_hvg_found = hvg_mask.sum()
        print(f"Found {n_hvg_found} HVGs, filtering data...")
        adata = adata[:, hvg_mask].copy()
        print(f"Filtered data: {adata.shape[0]} cells, {adata.shape[1]} HVG genes")
    
    # scPRINT embedding generation
    if use_scprint_embeddings:
        print("\n" + "="*50)
        print("GENERATING scPRINT EMBEDDINGS")
        print("="*50)
        
        try:
            from scprint import scPrint
            print("scPRINT imported successfully")
            
            # Load scPRINT model
            if scprint_model_path is not None:
                print(f"Loading scPRINT model from: {scprint_model_path}")
                model = scPrint.load_from_checkpoint(scprint_model_path, precpt_gene_emb=None)
            else:
                print("Loading pretrained scPRINT model...")
                # This might need adjustment based on scPRINT's API for pretrained models
                model = scPrint.load_from_pretrained()
            
            print("Model loaded successfully!")
            
            # CRITICAL: Use the already-filtered HVG data for scPRINT
            # Since we already filtered adata to HVGs above, use that
            adata_for_scprint = adata.copy()
            
            # Use raw counts for scPRINT (before normalization)
            if 'original_counts' in adata.layers:
                # Need to get original counts but only for the HVG genes
                original_hvg_indices = adata_for_scprint.var_names
                # Find these genes in the original data
                full_adata_temp = sc.AnnData(adata.layers['original_counts'], obs=adata.obs, var=adata.var)
                adata_for_scprint.X = full_adata_temp.X.copy()
            
            print(f"Using {adata_for_scprint.n_vars} HVG genes for scPRINT (pre-filtered)")
            
            # Get the list of HVG gene names that overlap with scPRINT's vocabulary
            hvg_gene_names = adata_for_scprint.var_names.tolist()
            scprint_compatible_genes = [g for g in hvg_gene_names if g in model.genes]
            
            print(f"HVGs available: {len(hvg_gene_names)} (target: {n_hvg})")
            print(f"HVGs compatible with scPRINT model: {len(scprint_compatible_genes)}")
            
            if len(scprint_compatible_genes) == 0:
                raise ValueError(f"No HVGs are compatible with scPRINT model vocabulary! "
                               f"Model knows {len(model.genes)} genes, but none overlap with your HVGs.")
            
            if len(scprint_compatible_genes) < n_hvg * 0.5:
                print(f"WARNING: Only {len(scprint_compatible_genes)} out of {n_hvg} "
                      f"HVGs are compatible with scPRINT model. Consider checking gene naming.")
            
            # Update the adata to only contain the compatible HVG genes
            compatible_gene_mask = adata_for_scprint.var_names.isin(scprint_compatible_genes)
            adata_for_scprint = adata_for_scprint[:, compatible_gene_mask].copy()
            
            print(f"Final dataset for scPRINT: {adata_for_scprint.n_obs} cells × {adata_for_scprint.n_vars} genes")
            
            # Check device availability
            if scprint_device == "cuda" and not torch.cuda.is_available():
                print("CUDA not available, switching to CPU")
                scprint_device = "cpu"
            
            print(f"Using device: {scprint_device}")
            print(f"Batch size: {scprint_batch_size}")
            
            # Extract gene embeddings ONLY for the HVG genes
            print(f"Extracting embeddings for {len(scprint_compatible_genes)} HVG genes...")
            gene_embeddings, genes_processed = extract_gene_embeddings(
                model=model,
                adata=adata_for_scprint,
                batch_size=scprint_batch_size,
                device=scprint_device,
                gene_list=scprint_compatible_genes  # Only use HVGs that scPRINT knows
            )
            
            print(f"Generated embeddings shape: {gene_embeddings.shape}")
            print(f"Embeddings: {gene_embeddings.shape[0]} cells × {gene_embeddings.shape[1]} genes × {gene_embeddings.shape[2]} dims")
            
            # Now we need to decide how to use these embeddings
            # Option 1: Average across genes to get cell embeddings
            # Option 2: Use gene-wise embeddings somehow
            # Option 3: Flatten to create a larger feature space
            
            # For CORTADO, I think we want to create a representation where each "gene" 
            # is now represented by its embedding features instead of expression
            
            # Let's reshape: (n_cells, n_genes * embedding_dim)
            n_cells, n_genes, emb_dim = gene_embeddings.shape
            
            # Flatten gene embeddings: each cell now has (n_genes * emb_dim) features
            flattened_embeddings = gene_embeddings.reshape(n_cells, n_genes * emb_dim)
            
            print(f"Flattened embeddings shape: {flattened_embeddings.shape}")
            
            # Create new feature names
            new_var_names = []
            for i, gene in enumerate(genes_processed):
                for j in range(emb_dim):
                    new_var_names.append(f"{gene}_emb_{j}")
            
            # Create new AnnData with embeddings
            adata_embedded = sc.AnnData(
                X=flattened_embeddings,
                obs=adata.obs.copy(),
                var=pd.DataFrame(index=new_var_names)
            )
            
            # Store metadata about embeddings
            adata_embedded.uns['scprint_embeddings'] = True
            adata_embedded.uns['scprint_genes_processed'] = genes_processed
            adata_embedded.uns['scprint_embedding_dim'] = emb_dim
            adata_embedded.uns['scprint_n_genes'] = n_genes
            adata_embedded.uns['scprint_n_hvg_requested'] = n_hvg
            adata_embedded.uns['scprint_n_hvg_used'] = len(genes_processed)
            
            # Store original data
            adata_embedded.layers['original_expression'] = adata.X.copy()
            adata_embedded.layers['original_counts'] = adata.layers['original_counts'].copy()
            
            print("scPRINT embedding generation complete!")
            print(f"✓ Used {len(genes_processed)} HVG genes (requested: {n_hvg})")
            print(f"✓ Generated {emb_dim}-dimensional embeddings per gene")
            print(f"✓ Final feature space: {adata_embedded.shape[1]} features ({n_genes} genes × {emb_dim} dims)")
            print("="*50)
            
            return adata_embedded
            
        except ImportError:
            print("ERROR: scPRINT not installed!")
            print("Please install with: pip install scprint")
            print("Falling back to original expression data...")
            
        except Exception as e:
            print(f"ERROR generating scPRINT embeddings: {str(e)}")
            print("Falling back to original expression data...")
            import traceback
            traceback.print_exc()
    
    # Mark that we're not using embeddings, but we still filtered to HVGs
    adata.uns['scprint_embeddings'] = False
    adata.uns['n_hvg_used'] = adata.n_vars
    
    print(f"Final data shape: {adata.shape[0]} cells, {adata.shape[1]} features (HVGs)")
    return adata