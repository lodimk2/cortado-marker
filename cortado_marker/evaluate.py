import pandas as pd
import matplotlib.pyplot as plt

def evaluate(best_solution, scores_df, true_markers_file, marker_column, plot_filepath='results_fig.png', csv_filepath='eval_results.csv', plot_title='Gene Selection Evaluation'):
    """
    Evaluate gene selection against true markers.

    Parameters:
    - best_solution (np.array): Best binary vector
    - scores_df (pd.DataFrame): Marker gene scores
    - true_markers_file (str): Path to true marker genes CSV file
    - marker_column (str): Column name in true marker CSV for markers
    - plot_filepath (str): Path to save evaluation plot
    - csv_filepath (str): Path to save evaluation results CSV
    - plot_title (str): Title for the evaluation plot

    Returns:
    - precision (float): Precision
    - recall (float): Recall
    - f1 (float): F1 score
    - accuracy (float): Accuracy
    """
    selected_indices = [i for i, x in enumerate(best_solution) if x == 1]

    # Subset the scaled_scores_df using the selected indices
    selected_genes = scores_df.iloc[selected_indices].index.tolist()
    # Load true marker genes from CSV
    true_marker_df = pd.read_csv(true_markers_file)

    # Extract the true marker genes for the given cluster label
    total_marker_genes = true_marker_df[marker_column].tolist()
    # Convert total_marker_genes to gene names assuming they match 'Gene_' format
    
    # FOR SIMULATED DATA ONLY, CHANGE THIS LATER 
    #total_marker_genes_names = [f'Gene_{gene}' for gene in total_marker_genes]

    # Convert lists to sets for calculations
    selected_genes_set = set(selected_genes)
    total_marker_genes_set = set(total_marker_genes)
    # Calculate True Positives, False Positives, False Negatives, and True Negatives
    TP = len(selected_genes_set & total_marker_genes_set)
    FP = len(selected_genes_set - total_marker_genes_set)
    FN = len(total_marker_genes_set - selected_genes_set)
    TN = 0  # TN is tricky in this context as it requires knowing all possible non-marker genes

    # Calculate Precision, Recall, F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate Accuracy
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    # Print results
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Save results to a CSV file
    results_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'Accuracy'],
        'Value': [precision, recall, f1, accuracy]
    })
    results_df.to_csv(csv_filepath, index=False)
    print(f"Evaluation metrics saved to {csv_filepath}")

    # Data for plotting
    metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values = [precision, recall, f1, accuracy]

    # Create bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'red', 'orange'])

    # Add labels and title
    plt.ylabel('Score')
    plt.title(plot_title)
    plt.ylim(0, 1)  # Set y-axis limits to 0-1

    # Show values on bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=12)

    # Save plot
    plt.savefig(plot_filepath)
    plt.close()
    print(f"Evaluation plot saved as {plot_filepath}")

    return precision, recall, f1, accuracy