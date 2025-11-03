#!/usr/bin/env python3
"""
Generate embedding visualizations for H-optimus-0 data.
Creates t-SNE, UMAP, and PCA plots colored by MSI status and site.
"""
import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

def load_hoptimus0_features_and_labels(feature_dir, clinical_table, slide_table):
    """Load H-optimus-0 feature embeddings and corresponding MSI labels."""
    print("Loading H-optimus-0 features and labels...")
    
    # Merge clinical and slide tables
    merged_data = pd.merge(slide_table, clinical_table, on='PATIENT', how='inner')
    
    features = []
    labels = []
    patients = []
    sites = []
    filenames = []
    
    missing_files = 0
    
    for idx, row in merged_data.iterrows():
        filename = row['FILENAME']
        msi_status = row['isMSIH']
        patient = row['PATIENT']
        site = row.get('SITE', 'Unknown')
        
        # Extract base filename for H5 file
        if isinstance(filename, str):
            # The filename might be a full path to H5 file or just the slide name
            if filename.endswith('.h5'):
                h5_path = filename  # Already a full path
            else:
                # Extract base filename without extension
                base_filename = os.path.basename(filename)
                base_filename = os.path.splitext(base_filename)[0]
                h5_path = os.path.join(feature_dir, f"{base_filename}.h5")
            
            if os.path.exists(h5_path):
                try:
                    with h5py.File(h5_path, 'r') as f:
                        # Try different possible keys for features in H5 file
                        feature_keys = ['features', 'embeddings', 'coords', 'patch_features', 'feats']
                        feature_data = None
                        
                        # Debug: print available keys
                        if idx < 3:  # Only for first few files
                            print(f"Available keys in {os.path.basename(h5_path)}: {list(f.keys())}")
                        
                        for key in feature_keys:
                            if key in f.keys():
                                feature_data = f[key][:]
                                if idx < 3:
                                    print(f"  Using key '{key}' with shape: {feature_data.shape}")
                                break
                        
                        if feature_data is not None:
                            # Take mean across patches to get slide-level representation
                            if len(feature_data.shape) > 1:
                                slide_embedding = np.mean(feature_data, axis=0)
                            else:
                                slide_embedding = feature_data
                            
                            features.append(slide_embedding)
                            labels.append(msi_status)
                            patients.append(patient)
                            sites.append(site)
                            filenames.append(os.path.basename(h5_path))
                        else:
                            print(f"Warning: No feature data found in {h5_path}")
                            print(f"  Available keys: {list(f.keys())}")
                            missing_files += 1
                            
                except Exception as e:
                    print(f"Error loading {h5_path}: {e}")
                    missing_files += 1
            else:
                print(f"File not found: {h5_path}")
                missing_files += 1
    
    print(f"Loaded {len(features)} slides with H-optimus-0 features")
    print(f"Missing {missing_files} feature files")
    
    if len(features) == 0:
        raise ValueError("No feature files could be loaded!")
    
    # Convert to numpy array
    features_array = np.array(features)
    print(f"Final feature array shape: {features_array.shape}")
    
    return features_array, labels, patients, sites, filenames

def create_embeddings(features, method='all', random_state=42):
    """Create various dimensionality reduction embeddings."""
    print(f"Creating embeddings using method: {method}")
    print(f"Input feature shape: {features.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    embeddings = {}
    
    if method in ['all', 'pca']:
        print("  Computing PCA...")
        pca = PCA(n_components=2, random_state=random_state)
        embeddings['PCA'] = pca.fit_transform(features_scaled)
        print(f"    PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"    Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    if method in ['all', 'tsne']:
        print("  Computing t-SNE...")
        # Use PCA preprocessing for t-SNE if features are high-dimensional
        if features_scaled.shape[1] > 50:
            pca_tsne = PCA(n_components=50, random_state=random_state)
            features_for_tsne = pca_tsne.fit_transform(features_scaled)
            print(f"    Using PCA preprocessing: {features_scaled.shape[1]} -> 50 dimensions")
        else:
            features_for_tsne = features_scaled
            
        perplexity = min(30, len(features)//4)
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
        embeddings['t-SNE'] = tsne.fit_transform(features_for_tsne)
        print(f"    Used perplexity: {perplexity}")
    
    if method in ['all', 'umap']:
        print("  Computing UMAP...")
        # Use PCA preprocessing for UMAP if features are high-dimensional
        if features_scaled.shape[1] > 50:
            pca_umap = PCA(n_components=50, random_state=random_state)
            features_for_umap = pca_umap.fit_transform(features_scaled)
            print(f"    Using PCA preprocessing: {features_scaled.shape[1]} -> 50 dimensions")
        else:
            features_for_umap = features_scaled
            
        n_neighbors = min(15, len(features)//3)
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors)
        embeddings['UMAP'] = reducer.fit_transform(features_for_umap)
        print(f"    Used n_neighbors: {n_neighbors}")
    
    return embeddings, scaler

def get_better_site_colors(sites):
    """Get better, more distinguishable colors for sites."""
    unique_sites = sorted(list(set(sites)))
    n_sites = len(unique_sites)
    
    # Use better color palettes based on number of sites
    if n_sites <= 8:
        # Use a high-contrast palette for fewer sites
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
                 '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    elif n_sites <= 12:
        # Use Set1 which has good contrast
        colors = sns.color_palette("Set1", n_sites)
    else:
        # For many sites, use a colorblind-friendly palette
        colors = sns.color_palette("tab20", n_sites)
    
    return dict(zip(unique_sites, colors))

def plot_embeddings(embeddings, labels, sites, patients, output_dir, model_name="H-optimus-0"):
    """Create visualization plots for embeddings."""
    print("Creating embedding plots...")
    
    # Color palettes
    msi_colors = {'MSI-H': '#FF5733', 'MSS': '#3366FF'}
    
    # Get better site colors
    site_colors = get_better_site_colors(sites)
    unique_sites = sorted(list(set(sites)))
    
    print(f"Number of unique sites: {len(unique_sites)}")
    print(f"Sites: {unique_sites}")
    
    # Create figure with subplots - MUCH BIGGER and SQUARE
    n_methods = len(embeddings)
    fig = plt.figure(figsize=(24, 8 * n_methods))  # Much wider and taller
    
    plot_idx = 1
    
    for method_name, embedding in embeddings.items():
        # Plot 1: Colored by MSI status
        plt.subplot(n_methods, 3, plot_idx)
        
        for msi_status in ['MSS', 'MSI-H']:  # Plot MSS first so MSI-H is on top
            mask = np.array(labels) == msi_status
            if np.any(mask):
                plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                          c=msi_colors[msi_status], label=msi_status, 
                          alpha=0.8, s=20, edgecolors='white', linewidth=0.2)  # Smaller dots, white edges
        
        plt.title(f'{method_name} - Colored by MSI Status ({model_name})', fontsize=16, pad=20)
        plt.xlabel(f'{method_name} 1', fontsize=14)
        plt.ylabel(f'{method_name} 2', fontsize=14)
        plt.legend(fontsize=12, markerscale=2)
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='box')  # Force square aspect ratio
        
        plot_idx += 1
        
        # Plot 2: IMPROVED site plot
        plt.subplot(n_methods, 3, plot_idx)
        
        # Plot sites in order of frequency (most common first, so rarer ones are on top)
        site_counts = pd.Series(sites).value_counts()
        sites_by_frequency = site_counts.index.tolist()
        
        for site in reversed(sites_by_frequency):  # Reverse so rare sites plot on top
            mask = np.array(sites) == site
            if np.any(mask):
                plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                          c=[site_colors[site]], label=f'{site} (n={site_counts[site]})', 
                          alpha=0.9, s=25, edgecolors='white', linewidth=0.3)
        
        plt.title(f'{method_name} - Colored by Site ({model_name})', fontsize=16, pad=20)
        plt.xlabel(f'{method_name} 1', fontsize=14)
        plt.ylabel(f'{method_name} 2', fontsize=14)
        
        # Improved legend
        if len(unique_sites) <= 8:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        else:
            # For many sites, create a more compact legend
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, 
                      ncol=1, markerscale=0.8)
        
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='box')  # Force square aspect ratio
        
        plot_idx += 1
        
        # Plot 3: MSI + Site combination (improved)
        plt.subplot(n_methods, 3, plot_idx)
        
        # Use different markers for sites, colors for MSI
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x', 
                  '8', 'P', 'X', 'd', '|', '_']
        site_markers = dict(zip(unique_sites, markers[:len(unique_sites)]))
        
        for msi_status in ['MSS', 'MSI-H']:
            for site in unique_sites:
                mask = (np.array(labels) == msi_status) & (np.array(sites) == site)
                if np.any(mask):
                    count = np.sum(mask)
                    plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                              c=msi_colors[msi_status], marker=site_markers[site],
                              alpha=0.9, s=30, edgecolors='white', linewidth=0.3,
                              label=f'{msi_status}-{site} (n={count})')
        
        plt.title(f'{method_name} - MSI Status + Site ({model_name})', fontsize=16, pad=20)
        plt.xlabel(f'{method_name} 1', fontsize=14)
        plt.ylabel(f'{method_name} 2', fontsize=14)
        
        # Compact legend for combination plot
        if len(unique_sites) <= 4:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, markerscale=1.2)
        else:
            # For many sites, make legend even more compact
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, 
                      ncol=1, markerscale=1.0)
        
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='box')  # Force square aspect ratio
        
        plot_idx += 1
    
    # Adjust layout with more space
    plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
    
    # Save with higher DPI for better quality
    plt.savefig(os.path.join(output_dir, f'embeddings_{model_name.replace("-", "_")}.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, f'embeddings_{model_name.replace("-", "_")}.pdf'), 
                bbox_inches='tight', facecolor='white')
    
    # Also create individual plots for each method (larger, cleaner)
    for method_name, embedding in embeddings.items():
        create_individual_plots(embedding, labels, sites, method_name, output_dir, model_name, msi_colors, site_colors)
    
    plt.close()
    
    print(f"Saved embedding plots to {output_dir}")

def create_individual_plots(embedding, labels, sites, method_name, output_dir, model_name, msi_colors, site_colors):
    """Create individual large plots for each embedding method."""
    
    unique_sites = sorted(list(set(sites)))
    site_counts = pd.Series(sites).value_counts()
    
    # Create a large figure for MSI status plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # Large square plot
    
    for msi_status in ['MSS', 'MSI-H']:
        mask = np.array(labels) == msi_status
        if np.any(mask):
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      c=msi_colors[msi_status], label=msi_status, 
                      alpha=0.7, s=15, edgecolors='white', linewidth=0.3)
    
    ax.set_title(f'{method_name} - MSI Status ({model_name})', fontsize=20, pad=20)
    ax.set_xlabel(f'{method_name} 1', fontsize=16)
    ax.set_ylabel(f'{method_name} 2', fontsize=16)
    ax.legend(fontsize=14, markerscale=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Style improvements
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_msi_status_{model_name.replace("-", "_")}.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_msi_status_{model_name.replace("-", "_")}.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create a large figure for site plot - IMPROVED
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))  # Slightly wider for legend
    
    # Plot in reverse frequency order so rare sites are on top
    sites_by_frequency = site_counts.index.tolist()
    
    for site in reversed(sites_by_frequency):
        mask = np.array(sites) == site
        if np.any(mask):
            count = np.sum(mask)
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      c=[site_colors[site]], label=f'{site} (n={count})', 
                      alpha=0.95, s=25, edgecolors='white', linewidth=0.4)
    
    ax.set_title(f'{method_name} - Sites ({model_name})', fontsize=20, pad=20)
    ax.set_xlabel(f'{method_name} 1', fontsize=16)
    ax.set_ylabel(f'{method_name} 2', fontsize=16)
    
    # Better legend positioning and formatting
    if len(unique_sites) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, 
                 markerscale=1.5, frameon=True, fancybox=True, shadow=True)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, 
                 markerscale=1.2, frameon=True, fancybox=True, shadow=True,
                 ncol=1)
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Style improvements
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_sites_{model_name.replace("-", "_")}.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_sites_{model_name.replace("-", "_")}.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    # BONUS: Create a site summary plot
    create_site_summary_plot(embedding, labels, sites, method_name, output_dir, 
                           model_name, msi_colors, site_colors)

def create_site_summary_plot(embedding, labels, sites, method_name, output_dir, 
                           model_name, msi_colors, site_colors):
    """Create a summary plot showing site distribution and MSI breakdown."""
    
    unique_sites = sorted(list(set(sites)))
    site_counts = pd.Series(sites).value_counts()
    
    # Create a figure with subplot for embedding and bar chart
    fig = plt.figure(figsize=(18, 8))
    
    # Main embedding plot
    ax1 = plt.subplot(1, 2, 1)
    
    for site in reversed(site_counts.index.tolist()):
        mask = np.array(sites) == site
        if np.any(mask):
            count = np.sum(mask)
            ax1.scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=[site_colors[site]], label=f'{site} (n={count})', 
                       alpha=0.9, s=30, edgecolors='white', linewidth=0.4)
    
    ax1.set_title(f'{method_name} - Sites ({model_name})', fontsize=16, pad=20)
    ax1.set_xlabel(f'{method_name} 1', fontsize=14)
    ax1.set_ylabel(f'{method_name} 2', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Site distribution bar chart
    ax2 = plt.subplot(1, 2, 2)
    
    # Create stacked bar chart showing MSI distribution within each site
    site_msi_data = []
    for site in site_counts.index:
        site_mask = np.array(sites) == site
        site_labels = np.array(labels)[site_mask]
        msi_h_count = np.sum(site_labels == 'MSI-H')
        mss_count = np.sum(site_labels == 'MSS')
        site_msi_data.append({'Site': site, 'MSI-H': msi_h_count, 'MSS': mss_count})
    
    site_df = pd.DataFrame(site_msi_data)
    site_df = site_df.set_index('Site')
    
    # Create stacked bar plot
    site_df.plot(kind='bar', stacked=True, ax=ax2, 
                color=[msi_colors['MSS'], msi_colors['MSI-H']], 
                alpha=0.8, edgecolor='white', linewidth=0.5)
    
    ax2.set_title(f'Site Distribution with MSI Status ({model_name})', fontsize=16, pad=20)
    ax2.set_xlabel('Site', fontsize=14)
    ax2.set_ylabel('Number of Samples', fontsize=14)
    ax2.legend(title='MSI Status', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add total counts on top of bars
    for i, (site, row) in enumerate(site_df.iterrows()):
        total = row.sum()
        ax2.text(i, total + 0.5, str(int(total)), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_site_summary_{model_name.replace("-", "_")}.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_site_summary_{model_name.replace("-", "_")}.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()

def analyze_clusters(embeddings, labels, sites, patients, output_dir, model_name="H-optimus-0"):
    """Perform clustering analysis on embeddings."""
    print("Performing clustering analysis...")
    
    results = {}
    
    for method_name, embedding in embeddings.items():
        print(f"  Analyzing {method_name}...")
        
        # Perform k-means clustering
        n_clusters = 2  # For MSI-H vs MSS
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding)
        
        # Calculate silhouette score
        silhouette = silhouette_score(embedding, cluster_labels)
        
        # Calculate ARI with true MSI labels
        msi_numeric = [1 if label == 'MSI-H' else 0 for label in labels]
        ari_msi = adjusted_rand_score(msi_numeric, cluster_labels)
        
        # Calculate ARI with site labels
        site_numeric = [list(set(sites)).index(site) for site in sites]
        ari_site = adjusted_rand_score(site_numeric, cluster_labels)
        
        results[method_name] = {
            'silhouette_score': silhouette,
            'ari_msi': ari_msi,
            'ari_site': ari_site,
            'cluster_labels': cluster_labels
        }
        
        print(f"    Silhouette Score: {silhouette:.3f}")
        print(f"    ARI (MSI): {ari_msi:.3f}")
        print(f"    ARI (Site): {ari_site:.3f}")
    
    # Save clustering results
    clustering_df = pd.DataFrame([
        {
            'Method': method,
            'Model': model_name,
            'Silhouette_Score': results[method]['silhouette_score'],
            'ARI_MSI': results[method]['ari_msi'],
            'ARI_Site': results[method]['ari_site']
        }
        for method in results.keys()
    ])
    
    clustering_df.to_csv(os.path.join(output_dir, f'clustering_analysis_{model_name.replace("-", "_")}.csv'), index=False)
    
    return results

def main():
    """Main function to run H-optimus-0 embedding visualization analysis."""
    
    # Configuration
    base_dir = '/lab/barcheese01/mdiberna/ARGO-DeepMSI'
    
    # Use the H-optimus-0 specific tables
    tables_dir = os.path.join(base_dir, 'tables', 'h_optimus_0')
    
    # Feature directory (consolidated H-optimus-0 features)
    feature_dir = os.path.join(base_dir, 'data', 'all', 'features', 'h-optimus-0')
    
    # Output directory
    output_dir = os.path.join(base_dir, 'visualizations', 'h_optimus_0', 'embeddings')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using tables from: {tables_dir}")
    print(f"Using features from: {feature_dir}")
    print(f"Saving results to: {output_dir}")
    
    # Load clinical and slide tables
    print("Loading clinical and slide tables...")
    clinical_table = pd.read_csv(os.path.join(tables_dir, 'all_clinical_table.csv'))
    slide_table = pd.read_csv(os.path.join(tables_dir, 'all_slide_table.csv'))
    
    print(f"Loaded {len(clinical_table)} patients and {len(slide_table)} slides")
    
    # Check MSI distribution
    msi_counts = clinical_table['isMSIH'].value_counts()
    print(f"MSI distribution: {msi_counts.to_dict()}")
    
    # Check site distribution
    if 'SITE' in slide_table.columns:
        site_counts = slide_table['SITE'].value_counts()
        print(f"Site distribution: {site_counts.to_dict()}")
    
    # Load features and labels
    try:
        features, labels, patients, sites, filenames = load_hoptimus0_features_and_labels(
            feature_dir, clinical_table, slide_table
        )
    except Exception as e:
        print(f"Error loading features: {e}")
        print("\nTroubleshooting:")
        print(f"1. Check if feature directory exists: {feature_dir}")
        print(f"2. Check if H5 files are present: ls {feature_dir}/*.h5")
        print(f"3. Verify table paths are correct")
        return
    
    print(f"\n{'='*50}")
    print(f"Processing H-optimus-0 embeddings")
    print(f"{'='*50}")
    
    # Create embeddings
    embeddings, scaler = create_embeddings(features, method='all')
    
    # Create visualizations
    plot_embeddings(embeddings, labels, sites, patients, output_dir, "H-optimus-0")
    
    # Analyze clusters
    clustering_results = analyze_clusters(embeddings, labels, sites, patients, output_dir, "H-optimus-0")
    
    # Create summary statistics
    summary_stats = []
    for method_name, embedding in embeddings.items():
        # MSI status separation
        msi_h_mask = np.array(labels) == 'MSI-H'
        mss_mask = np.array(labels) == 'MSS'
        
        if np.any(msi_h_mask) and np.any(mss_mask):
            msi_h_center = np.mean(embedding[msi_h_mask], axis=0)
            mss_center = np.mean(embedding[mss_mask], axis=0)
            separation = np.linalg.norm(msi_h_center - mss_center)
        else:
            separation = 0
        
        summary_stats.append({
            'Method': method_name,
            'Model': 'H-optimus-0',
            'Feature_Dim': features.shape[1],
            'N_Samples': len(features),
            'MSI_H_Count': sum(1 for l in labels if l == 'MSI-H'),
            'MSS_Count': sum(1 for l in labels if l == 'MSS'),
            'MSI_Separation': separation,
            'Silhouette_Score': clustering_results[method_name]['silhouette_score'],
            'ARI_MSI': clustering_results[method_name]['ari_msi']
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(output_dir, 'embedding_summary_h_optimus_0.csv'), index=False)
    
    print(f"\n{'='*50}")
    print("H-optimus-0 Embedding Analysis Complete!")
    print(f"{'='*50}")
    print(f"Results saved to: {output_dir}")
    print(f"Files created:")
    print(f"  - embeddings_H_optimus_0.png/pdf")
    print(f"  - clustering_analysis_H_optimus_0.csv")
    print(f"  - embedding_summary_h_optimus_0.csv")
    
    # Print summary
    print(f"\nSummary:")
    for _, row in summary_df.iterrows():
        print(f"  {row['Method']}: ARI_MSI = {row['ARI_MSI']:.3f}, Silhouette = {row['Silhouette_Score']:.3f}")

if __name__ == "__main__":
    main()