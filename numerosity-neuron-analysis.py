import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import argparse
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import time
import json
from datetime import datetime


def load_activations(model_dir, layer_name):
    """Load activations for a specific layer"""
    npz_file = os.path.join(model_dir, f"{layer_name}_activations.npz")
    
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"Activation file not found: {npz_file}")
    
    print(f"Loading activations from {npz_file}")
    data = np.load(npz_file)
    
    return {
        'activations': data['activations'],
        'numerosities': data['numerosities'],
        'stimulus_types': data['stimulus_types']
    }


def flatten_conv_layer(activations):
    """
    Flatten a convolutional layer completely to analyze each neuron independently
    
    Args:
        activations: Array of shape [n_samples, channels, height, width]
    
    Returns:
        Flattened array of shape [n_samples, channels*height*width]
    """
    if activations.ndim > 2:
        # Get original shape for reference
        orig_shape = activations.shape
        
        # Flatten to [n_samples, channels*height*width]
        n_samples = activations.shape[0]
        flat_activations = activations.reshape(n_samples, -1)
        
        print(f"Flattened activations from {orig_shape} to {flat_activations.shape}")
        return flat_activations
    else:
        # Already flat
        return activations


def find_numerosity_selective_neurons(activations, numerosities, stimulus_types, alpha=0.01, max_neurons=None):
    """
    Find neurons that are selective to numerosity but not to stimulus type
    
    Args:
        activations: Unit activations [n_samples, n_neurons]
        numerosities: Numerosity for each sample
        stimulus_types: Stimulus type for each sample
        alpha: Significance level for statistical tests
        max_neurons: Maximum number of neurons to analyze (for large layers)
    
    Returns:
        Dictionary with results for selective neurons
    """
    # Flatten activations if needed
    activations_flat = flatten_conv_layer(activations)
    
    n_neurons = activations_flat.shape[1]
    print(f"Total neurons to analyze: {n_neurons}")
    
    # If max_neurons is specified, randomly sample neurons
    if max_neurons is not None and n_neurons > max_neurons:
        print(f"Sampling {max_neurons} neurons for analysis")
        neuron_indices = np.random.choice(n_neurons, max_neurons, replace=False)
        neuron_indices = np.sort(neuron_indices)  # Sort for better organization
    else:
        neuron_indices = np.arange(n_neurons)
    
    # Storage for results
    selective_neurons = []
    preferred_numerosities = []
    neuron_indices_list = []
    p_values = []
    f_values = []
    
    # Prepare arrays for ANOVA
    numerosities_str = [str(n) for n in numerosities]
    
    # Perform 2-way ANOVA for each neuron
    start_time = time.time()
    for i, neuron_idx in enumerate(tqdm(neuron_indices, desc="Testing neurons")):
        neuron_acts = activations_flat[:, neuron_idx]
        
        # Create DataFrame for ANOVA
        data = pd.DataFrame({
            'activation': neuron_acts,
            'numerosity': numerosities_str,
            'stimulus_type': stimulus_types
        })
        
        try:
            # Fit the model and perform ANOVA
            formula = 'activation ~ C(numerosity) + C(stimulus_type) + C(numerosity):C(stimulus_type)'
            model = ols(formula, data=data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Get p-values
            p_numerosity = anova_table.loc['C(numerosity)', 'PR(>F)']
            p_stimulus = anova_table.loc['C(stimulus_type)', 'PR(>F)']
            p_interaction = anova_table.loc['C(numerosity):C(stimulus_type)', 'PR(>F)']
            
            # Progress update every 1000 neurons
            if i % 1000 == 0:
                elapsed = time.time() - start_time
                neurons_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(neuron_indices) - (i + 1)) / neurons_per_sec if neurons_per_sec > 0 else 0
                print(f"Processed {i+1}/{len(neuron_indices)} neurons "
                      f"({neurons_per_sec:.1f} neurons/sec, ETA: {eta/60:.1f} min)")
            
            # Check if neuron is selective for numerosity but not stimulus type
            if (p_numerosity < alpha) and (p_stimulus >= alpha) and (p_interaction >= alpha):
                # Find preferred numerosity
                avg_by_num = data.groupby('numerosity')['activation'].mean()
                preferred_num = float(avg_by_num.idxmax())
                
                selective_neurons.append(neuron_idx)
                neuron_indices_list.append(neuron_idx)
                preferred_numerosities.append(preferred_num)
                p_values.append(p_numerosity)
                f_values.append(anova_table.loc['C(numerosity)', 'F'])
        except Exception as e:
            # Skip neurons that cause errors in ANOVA
            print(f"Error processing neuron {neuron_idx}: {str(e)}")
    
    print(f"Analysis complete. Found {len(selective_neurons)} numerosity-selective neurons.")
    
    return {
        'selective_neurons': np.array(selective_neurons),
        'neuron_indices': np.array(neuron_indices_list),
        'preferred_numerosities': np.array(preferred_numerosities),
        'p_values': np.array(p_values),
        'f_values': np.array(f_values)
    }


def create_tuning_curves(activations, numerosities, stimulus_types, selective_neurons):
    """
    Create tuning curves for numerosity-selective neurons
    
    Args:
        activations: Neuron activations
        numerosities: Numerosity for each sample
        stimulus_types: Stimulus type for each sample
        selective_neurons: Indices of numerosity-selective neurons
    
    Returns:
        Dictionary with tuning curves
    """
    # Flatten activations if needed
    activations_flat = flatten_conv_layer(activations)
    
    # Get unique numerosities
    unique_nums = np.sort(np.unique(numerosities))
    
    # Create tuning curves for each selective neuron
    tuning_curves = {}
    normalized_tuning_curves = {}
    
    for neuron_idx in tqdm(selective_neurons, desc="Creating tuning curves"):
        neuron_acts = activations_flat[:, neuron_idx]
        
        # Calculate average response for each numerosity
        curve = []
        for num in unique_nums:
            mask = numerosities == num
            avg_response = np.mean(neuron_acts[mask])
            curve.append(avg_response)
        
        curve = np.array(curve)
        tuning_curves[neuron_idx] = curve
        
        # Normalize to [0, 1] for easier comparison
        if np.max(curve) > np.min(curve):
            norm_curve = (curve - np.min(curve)) / (np.max(curve) - np.min(curve))
        else:
            norm_curve = np.zeros_like(curve)
        normalized_tuning_curves[neuron_idx] = norm_curve
    
    return {
        'tuning_curves': tuning_curves,
        'normalized_tuning_curves': normalized_tuning_curves,
        'numerosities': unique_nums
    }


def analyze_tuning_width(tuning_curves, selective_neurons, preferred_numerosities, 
                         unique_nums, results_dir):
    """
    Analyze tuning width on different scales (linear, power, log)
    following methods in Nasr et al. 2019
    """
    from scipy.optimize import curve_fit
    
    # Define Gaussian function
    def gaussian(x, a, mu, sigma):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    # Define scales to test
    scales = {
        'linear': lambda x: x,
        'pow_0.5': lambda x: x**0.5,
        'pow_0.33': lambda x: x**0.33,
        'log2': lambda x: np.log2(x)
    }
    
    # Storage for results
    scale_r2_scores = {scale: [] for scale in scales}
    scale_sigmas = {scale: [] for scale in scales}
    
    # Counter for tracked neurons
    tracked_neurons = 0
    
    # Fit gaussian to each tuning curve on different scales
    for i, neuron_idx in enumerate(selective_neurons):
        # Only process a subset for performance in large datasets
        if i > 1000:  # Limit to 1000 neurons for curve fitting
            break
            
        curve = tuning_curves[neuron_idx]
        pref_num = preferred_numerosities[i]
        
        # Skip if preferred numerosity is at the edge of range
        if pref_num == min(unique_nums) or pref_num == max(unique_nums):
            continue
        
        tracked_neurons += 1
        
        for scale_name, scale_func in scales.items():
            try:
                # Transform x-axis
                x_scaled = scale_func(unique_nums)
                
                # Initial gaussian parameters (amplitude, mean, std)
                p0 = [1.0, scale_func(pref_num), 1.0]
                
                # Fit gaussian
                popt, _ = curve_fit(gaussian, x_scaled, curve, p0=p0)
                
                # Calculate fitted curve and R²
                curve_fit = gaussian(x_scaled, *popt)
                ss_res = np.sum((curve - curve_fit)**2)
                ss_tot = np.sum((curve - np.mean(curve))**2)
                r2 = 1 - (ss_res / ss_tot)
                
                # Store results
                scale_r2_scores[scale_name].append(r2)
                scale_sigmas[scale_name].append(popt[2])  # sigma
            except Exception as e:
                # Skip failures (poor fits)
                pass
    
    print(f"Tracked {tracked_neurons} neurons for tuning width analysis")
    
    # Convert lists to arrays
    for scale in scales:
        scale_r2_scores[scale] = np.array(scale_r2_scores[scale])
        scale_sigmas[scale] = np.array(scale_sigmas[scale])
    
    # Compare goodness of fit across scales
    plt.figure(figsize=(10, 6))
    labels = []
    r2_values = []
    r2_stds = []
    
    for scale in scales:
        if len(scale_r2_scores[scale]) > 0:
            labels.append(scale)
            r2_values.append(np.mean(scale_r2_scores[scale]))
            r2_stds.append(np.std(scale_r2_scores[scale]) / np.sqrt(len(scale_r2_scores[scale])))
    
    # Create bar plot with error bars
    plt.bar(labels, r2_values, yerr=r2_stds)
    plt.ylabel('Average R² (goodness of fit)', fontsize=14)
    plt.title('Gaussian Fit Quality on Different Scales', fontsize=16)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'goodness_of_fit_by_scale.png'))
    plt.close()
    
    # Plot sigma vs preferred numerosity for each scale
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    
    for i, (scale_name, scale_func) in enumerate(scales.items()):
        ax = axs[i]
        
        if len(scale_sigmas[scale_name]) == 0:
            ax.text(0.5, 0.5, "No data", horizontalalignment='center', fontsize=14)
            ax.set_title(f'Tuning Width on {scale_name} Scale', fontsize=14)
            continue
            
        sigmas = scale_sigmas[scale_name]
        
        # Get preferred numerosities for the neurons we have sigmas for
        prefs = []
        for j, neuron_idx in enumerate(selective_neurons):
            if j >= len(sigmas):
                break
            pref = preferred_numerosities[j]
            if pref != min(unique_nums) and pref != max(unique_nums):
                prefs.append(pref)
        
        prefs = prefs[:len(sigmas)]  # Match length with sigmas
        
        # Plot sigma vs preferred numerosity
        ax.scatter(prefs, sigmas, alpha=0.7)
        
        # Calculate correlation
        if len(prefs) > 0:
            corr, p_value = stats.pearsonr(prefs, sigmas)
            
            # Add trend line
            z = np.polyfit(prefs, sigmas, 1)
            p = np.poly1d(z)
            ax.plot(np.sort(prefs), p(np.sort(prefs)), 'r--', 
                    label=f'r={corr:.2f}, p={p_value:.3f}')
        
        ax.set_xlabel('Preferred Numerosity', fontsize=12)
        ax.set_ylabel('Tuning Width (sigma)', fontsize=12)
        ax.set_title(f'Tuning Width on {scale_name} Scale', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'tuning_width_by_scale.png'))
    plt.close()
    
    # Return summary of fitting results
    return {
        'scales': list(scales.keys()),
        'mean_r2': {scale: float(np.mean(vals)) if len(vals) > 0 else 0.0 
                   for scale, vals in scale_r2_scores.items()},
        'slope_linear': float(np.polyfit(prefs, scale_sigmas['linear'], 1)[0]) 
                       if len(scale_sigmas['linear']) > 0 else 0.0,
        'slope_log2': float(np.polyfit(prefs, scale_sigmas['log2'], 1)[0]) 
                     if len(scale_sigmas['log2']) > 0 else 0.0
    }


def visualize_selective_neurons(activations, numerosities, stimulus_types, 
                               selective_results, tuning_results, results_dir):
    """
    Create visualizations for numerosity-selective neurons
    
    Args:
        activations: Neuron activations
        numerosities: Numerosity for each sample
        stimulus_types: Stimulus type for each sample
        selective_results: Results from find_numerosity_selective_neurons
        tuning_results: Results from create_tuning_curves
        results_dir: Directory to save visualizations
    """
    # Get relevant data
    selective_neurons = selective_results['selective_neurons']
    preferred_numerosities = selective_results['preferred_numerosities']
    unique_nums = tuning_results['numerosities']
    norm_tuning_curves = tuning_results['normalized_tuning_curves']
    
    # Create directory for visualizations
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Visualize distribution of preferred numerosities
    plt.figure(figsize=(10, 6))
    plt.hist(preferred_numerosities, bins=len(np.unique(preferred_numerosities)))
    plt.xlabel('Preferred Numerosity', fontsize=14)
    plt.ylabel('Number of Neurons', fontsize=14)
    plt.title('Distribution of Preferred Numerosities', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'preferred_numerosity_distribution.png'))
    plt.close()
    
    # 2. Visualize tuning curves grouped by preferred numerosity
    # Group tuning curves by preferred numerosity
    numerosity_groups = {}
    for i, neuron_idx in enumerate(selective_neurons):
        pref_num = preferred_numerosities[i]
        if pref_num not in numerosity_groups:
            numerosity_groups[pref_num] = []
        numerosity_groups[pref_num].append(norm_tuning_curves[neuron_idx])
    
    # Plot average tuning curves for different preferred numerosities
    plt.figure(figsize=(12, 8))
    
    # Sort preferred numerosities and select a subset to avoid clutter
    preferred_nums = sorted(numerosity_groups.keys())
    step = max(1, len(preferred_nums) // 10)  # Show at most 10 curves
    
    for pref_num in preferred_nums[::step]:
        curves = numerosity_groups[pref_num]
        if len(curves) > 0:
            avg_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            
            plt.plot(unique_nums, avg_curve, 'o-', linewidth=2, label=f'Pref: {pref_num}')
            plt.fill_between(
                unique_nums, 
                np.maximum(0, avg_curve - std_curve), 
                np.minimum(1, avg_curve + std_curve), 
                alpha=0.2
            )
    
    plt.xlabel('Numerosity', fontsize=14)
    plt.ylabel('Normalized Response', fontsize=14)
    plt.title('Average Tuning Curves by Preferred Numerosity', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Preferred Numerosity')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'average_tuning_curves.png'))
    plt.close()
    
    # 3. Plot tuning curves on linear vs logarithmic scales
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Choose a subset of curves to visualize (to avoid clutter)
    n_selective = len(selective_neurons)
    if n_selective > 50:
        viz_indices = np.random.choice(n_selective, 50, replace=False)
    else:
        viz_indices = np.arange(n_selective)
    
    # Linear scale
    for i in viz_indices:
        neuron_idx = selective_neurons[i]
        pref_num = preferred_numerosities[i]
        curve = norm_tuning_curves[neuron_idx]
        ax1.plot(unique_nums, curve, 'o-', linewidth=1, alpha=0.5, label=f'{pref_num}')
    
    ax1.set_xlabel('Numerosity (Linear)', fontsize=12)
    ax1.set_ylabel('Normalized Response', fontsize=12)
    ax1.set_title('Tuning Curves on Linear Scale', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Logarithmic scale
    for i in viz_indices:
        neuron_idx = selective_neurons[i]
        pref_num = preferred_numerosities[i]
        curve = norm_tuning_curves[neuron_idx]
        ax2.plot(np.log2(unique_nums), curve, 'o-', linewidth=1, alpha=0.5, label=f'{pref_num}')
    
    ax2.set_xlabel('Numerosity (Log2 Scale)', fontsize=12)
    ax2.set_ylabel('Normalized Response', fontsize=12)
    ax2.set_title('Tuning Curves on Logarithmic Scale', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'tuning_curves_linear_vs_log.png'))
    plt.close()
    
    # 4. Create heatmap of all selective neuron responses
    # Take a subset for visualization if there are many neurons
    max_neurons_heatmap = 100
    if n_selective > max_neurons_heatmap:
        # Sample neurons from different preferred numerosities
        sampled_indices = []
        for pref_num in preferred_nums:
            indices = [i for i, p in enumerate(preferred_numerosities) if p == pref_num]
            n_sample = max(1, int(max_neurons_heatmap * len(indices) / n_selective))
            if len(indices) > 0:
                sampled_indices.extend(np.random.choice(indices, min(n_sample, len(indices)), replace=False))
        
        if len(sampled_indices) > max_neurons_heatmap:
            sampled_indices = np.random.choice(sampled_indices, max_neurons_heatmap, replace=False)
    else:
        sampled_indices = np.arange(n_selective)
    
    # Sort by preferred numerosity
    sorted_indices = sorted(sampled_indices, key=lambda i: preferred_numerosities[i])
    
    # Create heat map data
    heatmap_data = np.zeros((len(sorted_indices), len(unique_nums)))
    for i, idx in enumerate(sorted_indices):
        neuron_idx = selective_neurons[idx]
        heatmap_data[i] = norm_tuning_curves[neuron_idx]
    
    # Plot heatmap
    plt.figure(figsize=(10, 12))
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Normalized Response')
    plt.xlabel('Numerosity', fontsize=14)
    plt.ylabel('Neuron', fontsize=14)
    plt.title('Tuning of Numerosity-Selective Neurons', fontsize=16)
    plt.xticks(np.arange(len(unique_nums)), unique_nums)
    
    # Add preferred numerosity labels on y-axis for a subset of neurons
    step = max(1, len(sorted_indices) // 10)
    plt.yticks(np.arange(0, len(sorted_indices), step), 
              [f"N{i} (Pref: {preferred_numerosities[sorted_indices[i]]})" 
               for i in range(0, len(sorted_indices), step)])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'neuron_tuning_heatmap.png'))
    plt.close()


def analyze_and_visualize_selective_neurons(model_name, model_dir, output_dir, layer_name, alpha=0.01, max_neurons=None):
    """
    Analyze and visualize numerosity-selective neurons for a layer
    
    Args:
        model_name: Name of the model
        model_dir: Directory with saved activations
        output_dir: Directory to save analysis results
        layer_name: Name of the layer to analyze
        alpha: Significance level for statistical tests
        max_neurons: Maximum number of neurons to analyze (for large layers)
    """
    # Create output directory
    results_dir = os.path.join(output_dir, model_name, layer_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load activations
    print(f"Loading activations for {model_name} - {layer_name}")
    data = load_activations(model_dir, layer_name)
    activations = data['activations']
    numerosities = data['numerosities']
    stimulus_types = data['stimulus_types']
    
    # Find numerosity-selective neurons
    print(f"Finding numerosity-selective neurons for {model_name} - {layer_name}")
    selective_results = find_numerosity_selective_neurons(
        activations, numerosities, stimulus_types, alpha=alpha, max_neurons=max_neurons
    )
    selective_neurons = selective_results['selective_neurons']
    preferred_numerosities = selective_results['preferred_numerosities']
    
    # Count selective neurons
    n_total_neurons = np.prod(activations.shape[1:]) if activations.ndim > 2 else activations.shape[1]
    n_analyzed_neurons = len(selective_results['neuron_indices']) if max_neurons else n_total_neurons
    n_selective_neurons = len(selective_neurons)
    
    percentage_of_analyzed = (n_selective_neurons / n_analyzed_neurons) * 100
    percentage_of_total = (n_selective_neurons / n_total_neurons) * 100
    
    print(f"Found {n_selective_neurons} numerosity-selective neurons "
          f"({percentage_of_analyzed:.2f}% of analyzed, {percentage_of_total:.2f}% of total) in {layer_name}")
    
    # Save selective neuron information
    selective_df = pd.DataFrame({
        'neuron_idx': selective_neurons,
        'preferred_numerosity': preferred_numerosities,
        'p_value': selective_results['p_values'],
        'f_value': selective_results['f_values']
    })
    selective_df.to_csv(os.path.join(results_dir, 'selective_neurons.csv'), index=False)
    
    # Create summary file
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Layer: {layer_name}\n")
        f.write(f"Total neurons: {n_total_neurons}\n")
        f.write(f"Analyzed neurons: {n_analyzed_neurons}\n")
        f.write(f"Numerosity-selective neurons: {n_selective_neurons} ")
        f.write(f"({percentage_of_analyzed:.2f}% of analyzed, {percentage_of_total:.2f}% of total)\n")
        f.write(f"Statistical threshold (alpha): {alpha}\n\n")
        
        # Add distribution of preferred numerosities
        f.write("Distribution of preferred numerosities:\n")
        for num in sorted(np.unique(preferred_numerosities)):
            count = np.sum(preferred_numerosities == num)
            percent = (count / n_selective_neurons) * 100
            f.write(f"  Numerosity {num}: {count} neurons ({percent:.2f}%)\n")
    
    # If we have selective neurons, create tuning curves
    if n_selective_neurons > 0:
        # Calculate tuning curves
        print("Creating tuning curves for selective neurons")
        tuning_results = create_tuning_curves(
            activations, numerosities, stimulus_types, selective_neurons
        )
        
        # Visualize results
        print("Creating visualizations")
        visualize_selective_neurons(
            activations, numerosities, stimulus_types, 
            selective_results, tuning_results, results_dir
        )
        
        # Analyze tuning width
        print("Analyzing tuning width")
        tuning_width_results = analyze_tuning_width(
            tuning_results['normalized_tuning_curves'], selective_neurons, 
            preferred_numerosities, tuning_results['numerosities'], results_dir
        )
        
        # Save summary to JSON
        summary = {
            'model': model_name,
            'layer': layer_name,
            'total_neurons': int(n_total_neurons),
            'analyzed_neurons': int(n_analyzed_neurons),
            'selective_neurons': int(n_selective_neurons),
            'percentage_of_analyzed': float(percentage_of_analyzed),
            'percentage_of_total': float(percentage_of_total),
            'alpha': float(alpha),
            'preferred_numerosities': {
                str(num): int(np.sum(preferred_numerosities == num)) 
                for num in sorted(np.unique(preferred_numerosities))
            },
            'tuning_width': tuning_width_results
        }
        
        with open(os.path.join(results_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Analysis completed. Results saved to {results_dir}")
        return summary
    else:
        print(f"No numerosity-selective neurons found in {layer_name}")
        return None


def main():
    """Main function to run neuron-wise numerosity analysis"""
    parser = argparse.ArgumentParser(description="Analyze numerosity-selective neurons in neural networks")
    
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing model activation data')
    
    parser.add_argument('--output_dir', type=str, default='neuron_analysis',
                       help='Directory to save analysis results')
    
    parser.add_argument('--layer', type=str, required=True,
                       help='Layer name to analyze (e.g., "block1", "fc1")')
    
    parser.add_argument('--alpha', type=float, default=0.01,
                       help='Significance level for statistical tests')
    
    parser.add_argument('--model_name', type=str, default=None,
                       help='Optional model name (defaults to directory name)')
    
    parser.add_argument('--max_neurons', type=int, default=None,
                       help='Maximum number of neurons to analyze (useful for large layers)')
    
    parser.add_argument('--seed', type=int, default=21,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Set model name if not provided
    if args.model_name is None:
        args.model_name = os.path.basename(os.path.normpath(args.model_dir))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    config_file = os.path.join(args.output_dir, f"{args.model_name}_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Analyze the layer
    analyze_and_visualize_selective_neurons(
        model_name=args.model_name,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        layer_name=args.layer,
        alpha=args.alpha,
        max_neurons=args.max_neurons
    )
    
    print(f"Analysis configuration saved to {config_file}")


if __name__ == '__main__':
    main()
