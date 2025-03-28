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


def load_activations(model_dir, layer_name):
    """Load activations for a specific layer"""
    npz_file = os.path.join(model_dir, f"{layer_name}_activations.npz")
    data = np.load(npz_file)
    
    return {
        'activations': data['activations'],
        'numerosities': data['numerosities'],
        'stimulus_types': data['stimulus_types']
    }


def find_numerosity_selective_units(activations, numerosities, stimulus_types, alpha=0.01):
    """
    Find units that are selective to numerosity but not to stimulus type
    
    Args:
        activations: Unit activations [n_samples, n_units, ...]
        numerosities: Numerosity for each sample
        stimulus_types: Stimulus type for each sample
        alpha: Significance level for statistical tests
    
    Returns:
        List of selective unit indices and their preferred numerosities
    """
    # Flatten convolutional layers if needed
    if activations.ndim > 2:
        # Take mean across spatial dimensions for conv layers
        activations_flat = np.mean(activations, axis=(2, 3))
        
    else:
        activations_flat = activations
    
    n_units = activations_flat.shape[1]
    print(f"Analyzing {n_units} units")
    
    # Storage for results
    selective_units = []
    preferred_numerosities = []
    p_values = []
    f_values = []
    
    # Prepare arrays for ANOVA
    numerosities_str = [str(n) for n in numerosities]
    
    # Perform 2-way ANOVA for each unit
    for unit_idx in tqdm(range(n_units), desc="Testing units"):
        unit_acts = activations_flat[:, unit_idx]
        
        # Create DataFrame for ANOVA
        data = pd.DataFrame({
            'activation': unit_acts,
            'numerosity': numerosities_str,
            'stimulus_type': stimulus_types
        })
        
        try:
            # Fit the model and perform ANOVA
            formula = 'activation ~ C(numerosity) + C(stimulus_type) + C(numerosity):C(stimulus_type)'
            model = ols(formula, data=data).fit()
            #print(model.summary())
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Get p-values
            p_numerosity = anova_table.loc['C(numerosity)', 'PR(>F)']
            p_stimulus = anova_table.loc['C(stimulus_type)', 'PR(>F)']
            p_interaction = anova_table.loc['C(numerosity):C(stimulus_type)', 'PR(>F)']
            print(f"Unit {unit_idx}: p_numerosity={p_numerosity}, p_stimulus={p_stimulus}, p_interaction={p_interaction}")
            # Check if unit is selective for numerosity but not stimulus type
            if (p_numerosity < alpha) and (p_stimulus >= alpha) and (p_interaction >= alpha):
                # Find preferred numerosity
                avg_by_num = data.groupby('numerosity')['activation'].mean()
                preferred_num = float(avg_by_num.idxmax())
                
                selective_units.append(unit_idx)
                preferred_numerosities.append(preferred_num)
                p_values.append(p_numerosity)
                f_values.append(anova_table.loc['C(numerosity)', 'F'])
        except Exception as e:
            # Skip units that cause errors in ANOVA
            print(f"Error processing unit {unit_idx}: {str(e)}")
    
    return {
        'selective_units': np.array(selective_units),
        'preferred_numerosities': np.array(preferred_numerosities),
        'p_values': np.array(p_values),
        'f_values': np.array(f_values)
    }


def create_tuning_curves(activations, numerosities, stimulus_types, selective_units):
    """
    Create tuning curves for numerosity-selective units
    
    Args:
        activations: Unit activations
        numerosities: Numerosity for each sample
        stimulus_types: Stimulus type for each sample
        selective_units: Indices of numerosity-selective units
    
    Returns:
        Dictionary with tuning curves
    """
    # Flatten activations if needed
    if activations.ndim > 2:
        activations_flat = np.mean(activations, axis=(2, 3))
    else:
        activations_flat = activations
    
    # Get unique numerosities
    unique_nums = np.sort(np.unique(numerosities))
    
    # Create tuning curves for each selective unit
    tuning_curves = {}
    normalized_tuning_curves = {}
    
    for unit_idx in selective_units:
        unit_acts = activations_flat[:, unit_idx]
        
        # Calculate average response for each numerosity
        curve = []
        for num in unique_nums:
            mask = numerosities == num
            avg_response = np.mean(unit_acts[mask])
            curve.append(avg_response)
        
        curve = np.array(curve)
        tuning_curves[unit_idx] = curve
        
        # Normalize to [0, 1] for easier comparison
        norm_curve = (curve - np.min(curve)) / (np.max(curve) - np.min(curve))
        normalized_tuning_curves[unit_idx] = norm_curve
    
    return {
        'tuning_curves': tuning_curves,
        'normalized_tuning_curves': normalized_tuning_curves,
        'numerosities': unique_nums
    }


def analyze_and_visualize_selective_units(model_name, model_dir, output_dir, layer_name, alpha=0.01):
    """
    Analyze and visualize numerosity-selective units for a layer
    
    Args:
        model_name: Name of the model
        model_dir: Directory with saved activations
        output_dir: Directory to save analysis results
        layer_name: Name of the layer to analyze
        alpha: Significance level for statistical tests
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
    
    # Find numerosity-selective units
    print(f"Finding numerosity-selective units for {model_name} - {layer_name}")
    if layer_name=='block3':
        activations = np.reshape(activations, (activations.shape[0], -1))
    selective_results = find_numerosity_selective_units(
        activations, numerosities, stimulus_types, alpha=alpha
    )
    selective_units = selective_results['selective_units']
    preferred_numerosities = selective_results['preferred_numerosities']
    
    # Count selective units
    n_total_units = activations.shape[1]
    n_selective_units = len(selective_units)
    percentage = (n_selective_units / n_total_units) * 100
    
    print(f"Found {n_selective_units} numerosity-selective units ({percentage:.2f}%) in {layer_name}")
    
    # Save selective unit information
    selective_df = pd.DataFrame({
        'unit_idx': selective_units,
        'preferred_numerosity': preferred_numerosities,
        'p_value': selective_results['p_values'],
        'f_value': selective_results['f_values']
    })
    selective_df.to_csv(os.path.join(results_dir, 'selective_units.csv'), index=False)
    
    # Create summary file
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Layer: {layer_name}\n")
        f.write(f"Total units: {n_total_units}\n")
        f.write(f"Numerosity-selective units: {n_selective_units} ({percentage:.2f}%)\n")
        f.write(f"Statistical threshold (alpha): {alpha}\n\n")
        
        # Add distribution of preferred numerosities
        f.write("Distribution of preferred numerosities:\n")
        for num in sorted(np.unique(preferred_numerosities)):
            count = np.sum(preferred_numerosities == num)
            percent = (count / n_selective_units) * 100
            f.write(f"  Numerosity {num}: {count} units ({percent:.2f}%)\n")
    
    # If we have selective units, create tuning curves
    if n_selective_units > 0:
        # Calculate tuning curves
        tuning_results = create_tuning_curves(
            activations, numerosities, stimulus_types, selective_units
        )
        unique_nums = tuning_results['numerosities']
        norm_tuning_curves = tuning_results['normalized_tuning_curves']
        
        # Visualize distribution of preferred numerosities
        plt.figure(figsize=(10, 6))
        plt.hist(preferred_numerosities, bins=len(np.unique(preferred_numerosities)))
        plt.xlabel('Preferred Numerosity', fontsize=14)
        plt.ylabel('Number of Units', fontsize=14)
        plt.title(f'Distribution of Preferred Numerosities in {layer_name}', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'preferred_numerosity_distribution.png'))
        plt.close()
        
        # Visualize tuning curves grouped by preferred numerosity
        # Group tuning curves by preferred numerosity
        numerosity_groups = {}
        for unit_idx, pref_num in zip(selective_units, preferred_numerosities):
            if pref_num not in numerosity_groups:
                numerosity_groups[pref_num] = []
            numerosity_groups[pref_num].append(norm_tuning_curves[unit_idx])
        
        # Plot average tuning curves for different preferred numerosities
        plt.figure(figsize=(12, 8))
        for pref_num in sorted(numerosity_groups.keys()):
            curves = numerosity_groups[pref_num]
            avg_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            
            plt.plot(unique_nums, avg_curve, 'o-', linewidth=2, label=f'Pref: {pref_num}')
            plt.fill_between(
                unique_nums, 
                avg_curve - std_curve, 
                avg_curve + std_curve, 
                alpha=0.2
            )
        
        plt.xlabel('Numerosity', fontsize=14)
        plt.ylabel('Normalized Response', fontsize=14)
        plt.title(f'Average Tuning Curves by Preferred Numerosity in {layer_name}', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Preferred Numerosity')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'average_tuning_curves.png'))
        plt.close()
        
        # Plot tuning curves on linear vs logarithmic scales
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Choose a subset of curves to visualize (to avoid clutter)
        if n_selective_units > 50:
            viz_indices = np.random.choice(n_selective_units, 50, replace=False)
            viz_units = selective_units[viz_indices]
        else:
            viz_units = selective_units
        
        # Linear scale
        for unit_idx in viz_units:
            pref_num = preferred_numerosities[np.where(selective_units == unit_idx)[0][0]]
            curve = norm_tuning_curves[unit_idx]
            ax1.plot(unique_nums, curve, 'o-', linewidth=1, alpha=0.5, label=f'{pref_num}')
        
        ax1.set_xlabel('Numerosity (Linear)', fontsize=12)
        ax1.set_ylabel('Normalized Response', fontsize=12)
        ax1.set_title('Tuning Curves on Linear Scale', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Logarithmic scale
        for unit_idx in viz_units:
            pref_num = preferred_numerosities[np.where(selective_units == unit_idx)[0][0]]
            curve = norm_tuning_curves[unit_idx]
            ax2.plot(np.log2(unique_nums), curve, 'o-', linewidth=1, alpha=0.5, label=f'{pref_num}')
        
        ax2.set_xlabel('Numerosity (Log2 Scale)', fontsize=12)
        ax2.set_ylabel('Normalized Response', fontsize=12)
        ax2.set_title('Tuning Curves on Logarithmic Scale', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'tuning_curves_linear_vs_log.png'))
        plt.close()
        
        # Check if tuning is better on log scale by fitting Gaussians
        analyze_tuning_width(
            norm_tuning_curves, selective_units, preferred_numerosities, 
            unique_nums, results_dir
        )


def analyze_tuning_width(tuning_curves, selective_units, preferred_numerosities, 
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
    
    # Fit gaussian to each tuning curve on different scales
    for unit_idx in selective_units:
        curve = tuning_curves[unit_idx]
        pref_num = preferred_numerosities[np.where(selective_units == unit_idx)[0][0]]
        
        # Skip if preferred numerosity is at the edge of range
        if pref_num == min(unique_nums) or pref_num == max(unique_nums):
            continue
        
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
            except:
                # Skip failures (poor fits)
                pass
    
    # Convert lists to arrays
    for scale in scales:
        scale_r2_scores[scale] = np.array(scale_r2_scores[scale])
        scale_sigmas[scale] = np.array(scale_sigmas[scale])
    
    # Compare goodness of fit across scales
    plt.figure(figsize=(10, 6))
    labels = []
    r2_values = []
    for scale in scales:
        labels.append(scale)
        r2_values.append(np.mean(scale_r2_scores[scale]))
    
    plt.bar(labels, r2_values)
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
        sigmas = scale_sigmas[scale_name]
        
        # Get preferred numerosities for the units we have sigmas for
        # (some might have been skipped due to fitting failures)
        prefs = []
        for unit_idx in selective_units:
            pref = preferred_numerosities[np.where(selective_units == unit_idx)[0][0]]
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


def main():
    parser = argparse.ArgumentParser(description="Analyze numerosity-selective units in CNN")
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing model activation results')
    parser.add_argument('--output_dir', type=str, default='analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--model', type=str, choices=['simple', 'standard', 'both'], default='both',
                        help='Which model to analyze (simple=SimplerBallCounterCNN, standard=BallCounterCNN)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Significance level for statistical tests')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze models
    if args.model in ['standard', 'both']:
        model_name = 'BallCounterCNN_untrained'
        model_dir = os.path.join(args.results_dir, model_name)
        
        # Get available layers (except 'output')
        layers = [f.split('_')[0] for f in os.listdir(model_dir) 
                  if f.endswith('_activations.npz') and not f.startswith('output')]
        
        for layer_name in layers:
            print(f"\nAnalyzing {model_name} - {layer_name}")
            analyze_and_visualize_selective_units(
                model_name, model_dir, args.output_dir, 
                layer_name, alpha=args.alpha
            )
    
    if args.model in ['simple', 'both']:
        model_name = 'SimplerBallCounterCNN_untrained'
        model_dir = os.path.join(args.results_dir, model_name)
        
        # Get available layers (except 'output')
        layers = [f.split('_')[0] for f in os.listdir(model_dir) 
                  if f.endswith('_activations.npz') and not f.startswith('output')]
        
        for layer_name in layers:
            print(f"\nAnalyzing {model_name} - {layer_name}")
            analyze_and_visualize_selective_units(
                model_name, model_dir, args.output_dir, 
                layer_name, alpha=args.alpha
            )
    
    print("\nAnalysis complete! Results saved to", args.output_dir)


if __name__ == '__main__':
    main()
