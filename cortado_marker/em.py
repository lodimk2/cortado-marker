# cortado_marker/em.py

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
from typing import Optional, List, Tuple
import cortado_marker as cortado

class CORTADO_EM:
    """
    Expectation-Maximization wrapper for CORTADO marker selection.
    Iteratively refines clustering and marker selection.
    """
    
    def __init__(
        self,
        max_iterations: int = 10,
        convergence_threshold: float = 0.95,
        n_markers_per_cluster: int = 10,
        verbose: bool = True
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.n_markers_per_cluster = n_markers_per_cluster
        self.verbose = verbose
        
        # Store history for analysis
        self.history = {
            'markers': [],
            'n_clusters': [],
            'ari_scores': [],
            'marker_stability': []
        }
    
    def fit(
        self, 
        adata,
        initial_markers: Optional[List[str]] = None,
        resolution: float = 1.0
    ) -> Tuple[List[str], np.ndarray]:
        """
        Main EM loop: alternate between clustering (E-step) and marker selection (M-step)
        
        Args:
            adata: AnnData object with expression data
            initial_markers: Starting marker genes (if None, use all genes)
            resolution: Leiden clustering resolution
            
        Returns:
            final_markers: List of selected marker genes
            final_clusters: Final cluster assignments
        """
        
        # Initialize with all genes or provided markers
        if initial_markers is None:
            # Start with highly variable genes for efficiency
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, inplace=True)
            current_markers = adata.var_names[adata.var.highly_variable].tolist()
            if self.verbose:
                print(f"Starting with {len(current_markers)} highly variable genes")
        else:
            current_markers = initial_markers
        
        prev_clusters = None
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
            
            # ============ E-STEP: Cluster with current markers ============
            current_clusters = self._clustering_step(
                adata, current_markers, resolution
            )
            
            # Check convergence
            if prev_clusters is not None:
                ari = adjusted_rand_score(prev_clusters, current_clusters)
                self.history['ari_scores'].append(ari)
                
                if self.verbose:
                    print(f"  ARI with previous clustering: {ari:.3f}")
                
                if ari >= self.convergence_threshold:
                    if self.verbose:
                        print(f"  ✓ Converged! (ARI >= {self.convergence_threshold})")
                    break
            
            # ============ M-STEP: Select markers given clusters ============
            new_markers = self._marker_selection_step(
                adata, current_clusters
            )
            
            # Calculate marker stability
            if iteration > 0:
                marker_overlap = len(set(current_markers) & set(new_markers))
                stability = marker_overlap / len(new_markers)
                self.history['marker_stability'].append(stability)
                
                if self.verbose:
                    print(f"  Marker stability: {stability:.2%} ({marker_overlap}/{len(new_markers)} genes)")
            
            # Update for next iteration
            self.history['markers'].append(new_markers)
            self.history['n_clusters'].append(len(np.unique(current_clusters)))
            current_markers = new_markers
            prev_clusters = current_clusters
        
        # Store final results
        self.final_markers = current_markers
        self.final_clusters = current_clusters
        
        return current_markers, current_clusters
    
    def _clustering_step(
        self, 
        adata, 
        markers: List[str],
        resolution: float
    ) -> np.ndarray:
        """
        E-step: Perform clustering using current marker genes
        """
        # Subset to marker genes
        adata_subset = adata[:, markers].copy()
        
        # Standard preprocessing
        sc.pp.normalize_total(adata_subset, target_sum=1e4)
        sc.pp.log1p(adata_subset)
        sc.pp.scale(adata_subset, max_value=10)
        
        # PCA and clustering
        n_comps = min(50, len(markers) - 1)
        sc.tl.pca(adata_subset, n_comps=n_comps)
        sc.pp.neighbors(adata_subset, n_neighbors=30, n_pcs=n_comps)
        sc.tl.leiden(adata_subset, resolution=resolution)
        
        clusters = adata_subset.obs['leiden'].values.astype(int)
        
        if self.verbose:
            print(f"  E-step: Found {len(np.unique(clusters))} clusters using {len(markers)} markers")
        
        return clusters
    
    def _marker_selection_step(self, adata, clusters: np.ndarray) -> List[str]:
        """
        M-step: Select markers using CORTADO for each cluster
        """
        import numpy as np
        import pandas as pd
        import cortado_marker as cortado

        # 1) Put EM clusters into a column CORTADO definitely knows (fallback)
        #    Also standardize to string categories to keep comparisons simple.
        adata = adata.copy()  # avoid views
        em_name = "em_clusters"
        adata.obs[em_name] = pd.Categorical(clusters.astype(str))
        # Fallback/compat: many functions expect 'clust_assign'
        adata.obs["clust_assign"] = adata.obs[em_name]

        all_markers = []
        min_cells = 15  # skip tiny clusters to avoid unstable DE

        for cluster_id in np.unique(clusters):
            if self.verbose:
                print(f"  M-step: Selecting markers for cluster {cluster_id}...", end="")

            # Ensure we're selecting by the SAME string representation
            cid = str(cluster_id)
            n_in = (adata.obs[em_name] == cid).sum()
            n_out = adata.n_obs - n_in
            if n_in < min_cells or n_out < min_cells:
                if self.verbose:
                    print(f" Skipped (too few cells: in={n_in}, out={n_out})")
                continue

            try:
                # ---- Step 1: DE prefilter (avoid use_raw=True unless you know adata.raw is raw counts) ----
                marker_scores = cortado.calc_marker_gene_score(
                    adata,
                    target_cluster=cid,
                    n_genes=50,           # candidate cap
                    p_val_threshold=0.10, # relax a bit to avoid empty sets
                    use_raw=False,        # safer default given your log warning
                    # If your branch accepts cluster_column, keep it:
                    # cluster_column=em_name
                )

                # Normalize to DF with the expected column
                if isinstance(marker_scores, pd.Series):
                    marker_scores = pd.DataFrame({"marker_score": marker_scores})
                elif isinstance(marker_scores, pd.DataFrame):
                    if marker_scores.shape[1] == 1 and "marker_score" not in marker_scores.columns:
                        marker_scores = marker_scores.rename(columns={marker_scores.columns[0]: "marker_score"})
                    elif "marker_score" not in marker_scores.columns:
                        # If your scorer returns multiple columns, pick/compute the main one here:
                        # e.g., marker_scores["marker_score"] = marker_scores["score"] or similar
                        raise ValueError("marker_scores DF must include 'marker_score' column.")
                else:
                    raise TypeError("marker_scores must be a pandas Series or DataFrame")

                # Drop NaN/inf and sort
                ms_before = len(marker_scores)
                marker_scores = marker_scores.replace([np.inf, -np.inf], np.nan).dropna(subset=["marker_score"])
                marker_scores = marker_scores.sort_values("marker_score", ascending=False)

                if marker_scores.empty:
                    if self.verbose:
                        print(f" Failed: no DE genes (after filtering from {ms_before})")
                    continue

                # ---- Step 2: Correlation/similarity on same cluster label ----
                sim_scores = cortado.gene_correlation_within_cluster(
                    cid,
                    adata,
                    # cluster_column=em_name  # if supported
                )
                # Filter to candidate genes + enforce order
                filtered_genes = marker_scores.index
                sim_idx = sim_scores.index
                missing = [g for g in filtered_genes if g not in sim_idx]
                if missing and self.verbose:
                    print(f" (warn: {len(missing)} genes missing in corr) ", end="")
                filtered_corr_matrix = sim_scores.loc[filtered_genes.intersection(sim_idx), filtered_genes.intersection(sim_idx)]
                # Re-align marker_scores to what remains in the corr matrix
                kept = filtered_corr_matrix.index
                marker_scores = marker_scores.loc[kept]
                # Final alignment check
                if marker_scores.empty or filtered_corr_matrix.empty:
                    if self.verbose:
                        print(" Failed: empty after corr alignment")
                    continue

                # ---- Step 3: Hill-climbing with safe how_many ----
                nGenes = len(marker_scores)
                how_many = min(self.n_markers_per_cluster, nGenes)
                if how_many == 0:
                    if self.verbose:
                        print(" Failed: how_many==0")
                    continue

                best_solution, best_value = cortado.run_stochastic_hill_climbing(
                    marker_scores,                # DF with 'marker_score'
                    filtered_corr_matrix,         # square DF aligned to marker_scores.index
                    how_many=how_many,
                    max_iterations=300,           # tweak as you like
                    gamma=0.7,
                    idle_limit=20,
                    lambda1=0.9,
                    lambda2=0.1,
                    lambda3=0.0,
                    mode=1,
                    plot_filename=""
                )

                sel = np.asarray(best_solution).reshape(-1)
                cluster_markers = marker_scores.index[sel == 1].tolist()
                all_markers.extend(cluster_markers)

                if self.verbose:
                    print(f" {len(cluster_markers)} markers selected")

            except Exception as e:
                if self.verbose:
                    print(f" Failed: {e}")
                continue

        # Deduplicate preserving order
        unique_markers = list(dict.fromkeys(all_markers))
        if self.verbose:
            print(f"  M-step: Total {len(unique_markers)} unique markers selected")
        return unique_markers


def run_cortado_em(
    adata,
    n_iterations: int = 5,
    n_markers: int = 10,
    resolution: float = 1.0,
    random_state: int = 42
) -> Tuple[List[str], np.ndarray]:
    """
    Simple wrapper function for quick testing
    
    Example:
        markers, clusters = run_cortado_em(adata, n_iterations=5)
    """
    np.random.seed(random_state)
    
    em = CORTADO_EM(
        max_iterations=n_iterations,
        n_markers_per_cluster=n_markers,
        verbose=True
    )
    
    markers, clusters = em.fit(adata, resolution=resolution)
    
    # Add results to adata
    adata.obs['cortado_em_clusters'] = pd.Categorical(clusters)
    adata.var['cortado_em_markers'] = adata.var_names.isin(markers)
    
    return markers, clusters