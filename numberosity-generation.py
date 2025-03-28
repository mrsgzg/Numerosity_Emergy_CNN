import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from collections import defaultdict
import json
import time
import random
from datetime import datetime

# Import the models
from Models import (
    SingleLayerCNN, TwoLayerCNN, ThreeLayerCNN, FourLayerCNN, AlexNetCNN, 
    SingleLayerMLP, TwoLayerMLP, ThreeLayerMLP, FourLayerMLP, FiveLayerMLP
)


class NumerosityDataset(Dataset):
    """Dataset for loading numerosity images"""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file: Path to CSV file with image paths and metadata
            root_dir: Directory containing the images
            transform: Optional transform to be applied to images
        """
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path (adjust path if needed)
        img_path = self.data_info.iloc[idx]['image_path']
        if not os.path.isabs(img_path):
            # Try to find the image - it might be in a subdirectory
            base_name = os.path.basename(img_path)
            stim_type = self.data_info.iloc[idx]['stimulus_type']
            img_path = os.path.join(self.root_dir, stim_type, base_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(self.root_dir, base_name)
        
        # Load image as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure image is loaded
        if image is None:
            raise ValueError(f"Failed to load image at {img_path}")
        
        # Get numerosity and stimulus type
        numerosity = self.data_info.iloc[idx]['numerosity']
        stimulus_type = self.data_info.iloc[idx]['stimulus_type']
        
        # Convert to tensor and normalize to [0, 1]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image).unsqueeze(0) / 255.0
        
        return image, numerosity, stimulus_type


def set_random_seeds(seed, cuda_deterministic=True):
    """
    Sets random seeds for reproducibility
    
    Args:
        seed: Random seed to use
        cuda_deterministic: If True, makes CUDA operations deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_weights(model, init_method='xavier_uniform', gain=1.0, mean=0.0, std=0.05, min_val=-0.05, max_val=0.05):
    """
    Initialize model weights using specified method
    
    Args:
        model: The neural network model
        init_method: Weight initialization method
        gain: Gain parameter for some initialization methods
        mean: Mean for normal initialization
        std: Standard deviation for normal initialization
        min_val: Minimum value for uniform initialization
        max_val: Maximum value for uniform initialization
    """
    import torch.nn.init as init
    import torch.nn as nn
    
    def init_fn(module):
        # Apply initialization only to convolutional and linear layers
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if init_method == 'xavier_uniform':
                init.xavier_uniform_(module.weight, gain=gain)
            elif init_method == 'xavier_normal':
                init.xavier_normal_(module.weight, gain=gain)
            elif init_method == 'kaiming_uniform':
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            elif init_method == 'kaiming_normal':
                init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            elif init_method == 'orthogonal':
                init.orthogonal_(module.weight, gain=gain)
            elif init_method == 'normal':
                init.normal_(module.weight, mean=mean, std=std)
            elif init_method == 'uniform':
                init.uniform_(module.weight, a=min_val, b=max_val)
            else:
                print(f"Warning: Unknown initialization method: {init_method}, using xavier_uniform")
                init.xavier_uniform_(module.weight, gain=gain)
                
            # Initialize bias if it exists
            if module.bias is not None:
                init.zeros_(module.bias)
                
        # Add specific initializations for other layer types if needed
        # For example, BatchNorm layers
        elif isinstance(module, nn.BatchNorm2d):
            if module.weight is not None:
                init.ones_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
    
    # Apply the initialization function to all modules
    model.apply(init_fn)
    
    print(f"Model initialized with {init_method} method")


def extract_activations(model, dataloader, device, save_dir=None, layer_batch_size=None):
    """
    Extract activations from all layers of the model for the given stimuli
    
    Args:
        model: Neural network model
        dataloader: DataLoader with numerosity stimuli
        device: Device to use for computation
        save_dir: If provided, save activations incrementally to this directory
        layer_batch_size: If provided, save activations for each layer after this many batches
    
    Returns:
        Dictionary containing activations and metadata
    """
    # Storage for activations and metadata
    all_activations = defaultdict(list)
    all_numerosities = []
    all_stimulus_types = []
    
    # Set model to evaluation mode
    model.eval()
    
    # Create save directory if needed
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Initialize counters for incremental saving
    batch_counter = 0
    save_counter = 0
    
    # Process all batches
    with torch.no_grad():
        for images, numerosities, stimulus_types in tqdm(dataloader, desc="Extracting activations"):
            # Move images to device
            images = images.to(device)
            
            # Forward pass to get features from all layers
            features = model(images, return_features=True)
            
            # Store activations from each layer
            for layer_name, activations in features.items():
                # Convert to numpy and store
                act_np = activations.cpu().numpy()
                all_activations[layer_name].append(act_np)
            
            # Store metadata
            all_numerosities.extend(numerosities.numpy())
            all_stimulus_types.extend(stimulus_types)
            
            # Increment batch counter
            batch_counter += 1
            
            # Save incrementally if requested
            if save_dir is not None and layer_batch_size is not None and batch_counter % layer_batch_size == 0:
                print(f"Saving incremental batch {save_counter+1} (processed {batch_counter} batches)")
                temp_results = {
                    'activations': {layer: np.concatenate(all_activations[layer], axis=0) for layer in all_activations},
                    'numerosities': np.array(all_numerosities),
                    'stimulus_types': np.array(all_stimulus_types)
                }
                
                # Print memory usage stats
                for layer_name, layer_act in temp_results['activations'].items():
                    mem_mb = layer_act.nbytes / (1024 * 1024)
                    print(f"  Layer {layer_name}: Shape {layer_act.shape}, Memory {mem_mb:.2f} MB")
                
                save_activations(temp_results, save_dir, f"incremental_{save_counter}")
                save_counter += 1
                
                # Clear memory
                all_activations = defaultdict(list)
                all_numerosities = []
                all_stimulus_types = []
    
    # Final save or return
    if save_dir is not None and (layer_batch_size is None or batch_counter % layer_batch_size != 0):
        # Concatenate batch results
        for layer_name in all_activations:
            all_activations[layer_name] = np.concatenate(all_activations[layer_name], axis=0)
        
        # Convert lists to arrays
        all_numerosities = np.array(all_numerosities)
        all_stimulus_types = np.array(all_stimulus_types)
        
        # Save final results
        final_results = {
            'activations': all_activations,
            'numerosities': all_numerosities,
            'stimulus_types': all_stimulus_types
        }
        
        save_activations(final_results, save_dir, f"final")
        return final_results
    
    # If not saving incrementally, concatenate everything
    else:
        print("Processing complete, concatenating results...")
        # Concatenate batch results
        for layer_name in all_activations:
            print(f"  Concatenating layer {layer_name}")
            all_activations[layer_name] = np.concatenate(all_activations[layer_name], axis=0)
            # Print shape info
            shape = all_activations[layer_name].shape
            mem_mb = all_activations[layer_name].nbytes / (1024 * 1024)
            print(f"    Shape: {shape}, Memory: {mem_mb:.2f} MB")
        
        # Convert lists to arrays
        all_numerosities = np.array(all_numerosities)
        all_stimulus_types = np.array(all_stimulus_types)
        
        return {
            'activations': all_activations,
            'numerosities': all_numerosities,
            'stimulus_types': all_stimulus_types
        }


def save_model_weights(model, output_dir, model_name, init_method=None, seed=None):
    """
    Save model weights for reproducibility
    
    Args:
        model: Neural network model
        output_dir: Directory to save the weights
        model_name: Name of the model (for subdirectory)
        init_method: Weight initialization method used (for metadata)
        seed: Random seed used (for metadata)
    """
    # Create output directory for model weights
    weights_dir = os.path.join(output_dir, model_name, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model state dict
    model_path = os.path.join(weights_dir, f"{model_name}_weights_{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'architecture': str(model.__class__.__name__),
        'init_method': init_method,
        'random_seed': seed,
        'weights_file': os.path.basename(model_path)
    }
    
    with open(os.path.join(weights_dir, f"{model_name}_metadata_{timestamp}.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model weights saved to {model_path}")


def save_activations(results, output_dir, model_name):
    """
    Save extracted activations and metadata to disk
    
    Args:
        results: Dictionary with activations and metadata
        output_dir: Directory to save results
        model_name: Name of the model (for subdirectory)
    """
    # Create output directory for this model
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Get activations and metadata
    activations = results['activations']
    numerosities = results['numerosities']
    stimulus_types = results['stimulus_types']
    
    # Save each layer's activations separately to avoid enormous files
    for layer_name, layer_activations in activations.items():
        layer_file = os.path.join(model_dir, f"{layer_name}_activations.npz")
        np.savez_compressed(
            layer_file,
            activations=layer_activations,
            numerosities=numerosities,
            stimulus_types=stimulus_types,
            timestamp=np.array([time.time()])  # Add timestamp for tracking
        )
        print(f"Saved {layer_name} activations to {layer_file}")
    
    # Save a summary file with shapes and metadata
    with open(os.path.join(model_dir, 'summary.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total samples: {len(numerosities)}\n")
        f.write(f"Unique numerosities: {sorted(np.unique(numerosities))}\n")
        f.write(f"Unique stimulus types: {sorted(np.unique(stimulus_types))}\n\n")
        
        f.write("Layer shapes:\n")
        for layer_name, acts in activations.items():
            f.write(f"  {layer_name}: {acts.shape}\n")

    # Create a metadata file in JSON format for better structured data
    metadata = {
        'model_name': model_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_samples': len(numerosities),
        'unique_numerosities': [int(n) for n in sorted(np.unique(numerosities))],
        'unique_stimulus_types': sorted(np.unique(stimulus_types).tolist()),
        'layer_shapes': {layer_name: list(acts.shape) for layer_name, acts in activations.items()}
    }
    
    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)


def create_basic_visualizations(results, output_dir, model_name):
    """
    Create basic visualizations of activations by numerosity
    
    Args:
        results: Dictionary with activations and metadata
        output_dir: Directory to save results
        model_name: Name of the model (for subdirectory)
    """
    # Create visualization directory for this model
    vis_dir = os.path.join(output_dir, model_name, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get activations and metadata
    activations = results['activations']
    numerosities = results['numerosities']
    stimulus_types = results['stimulus_types']
    
    # Get unique numerosities and stimulus types
    unique_nums = np.sort(np.unique(numerosities))
    unique_stims = np.unique(stimulus_types)
    
    # For each layer, visualize response patterns to numerosity
    for layer_name, layer_acts in activations.items():
        # Skip output layer
        if layer_name == 'output':
            continue
        
        print(f"Creating visualizations for layer: {layer_name} with shape {layer_acts.shape}")
        
        # Properly reshape convolutional layers
        if layer_acts.ndim > 2:
            # Reshape to [batch_size, total_units]
            orig_shape = layer_acts.shape
            batch_size = layer_acts.shape[0]
            layer_acts_flat = layer_acts.reshape(batch_size, -1)
            print(f"  Reshaped from {orig_shape} to {layer_acts_flat.shape}")
            
            # For convolutional layers, we'll also visualize a few channel maps
            # Just for reference, since we can't visualize all flattened units
            if layer_name != 'output':
                # Select a few random channels to visualize
                num_channels = min(5, layer_acts.shape[1])
                random_channels = np.random.choice(layer_acts.shape[1], num_channels, replace=False)
                
                for channel_idx in random_channels:
                    channel_acts = layer_acts[:, channel_idx, :, :]
                    # Flatten spatial dimensions only
                    channel_acts_flat = channel_acts.reshape(batch_size, -1)
                    
                    # Calculate average response by numerosity for each spatial position
                    spatial_positions = channel_acts_flat.shape[1]
                    # Limit to 10 random spatial positions if there are too many
                    if spatial_positions > 10:
                        selected_positions = np.random.choice(spatial_positions, 10, replace=False)
                    else:
                        selected_positions = range(spatial_positions)
                    
                    plt.figure(figsize=(15, 10))
                    for pos_idx in selected_positions:
                        pos_acts = channel_acts_flat[:, pos_idx]
                        
                        # Calculate average response by numerosity
                        avg_by_num = []
                        for num in unique_nums:
                            num_mask = numerosities == num
                            avg_by_num.append(np.mean(pos_acts[num_mask]))
                        
                        plt.plot(unique_nums, avg_by_num, marker='o', linewidth=1.5, 
                                label=f'Position {pos_idx}')
                    
                    plt.xlabel('Numerosity', fontsize=14)
                    plt.ylabel('Average Activation', fontsize=14)
                    plt.title(f'{layer_name} - Channel {channel_idx} Spatial Positions Response', fontsize=16)
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='best')
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f'{layer_name}_channel{channel_idx}_spatial_response.png'))
                    plt.close()
        else:
            layer_acts_flat = layer_acts
        
        # Get total number of units
        n_units = layer_acts_flat.shape[1]
        
        # For large layers, select more diverse units: some random, some with highest variance
        if n_units > 50:
            # Calculate variance of each unit across all samples
            unit_variances = np.var(layer_acts_flat, axis=0)
            
            # Get indices of units with highest variance
            high_var_indices = np.argsort(unit_variances)[-10:]
            
            # Get random indices for the rest
            random_indices = np.random.choice(
                [i for i in range(n_units) if i not in high_var_indices],
                min(10, n_units-10), 
                replace=False
            )
            
            # Combine both sets
            selected_units = np.concatenate([high_var_indices, random_indices])
        else:
            # For smaller layers, select a subset of units randomly
            selected_units = np.random.choice(n_units, min(20, n_units), replace=False)
        
        # Create a figure for overall numerosity response
        plt.figure(figsize=(15, 10))
        
        # Plot average response by numerosity for selected units
        for unit_idx in selected_units:
            unit_acts = layer_acts_flat[:, unit_idx]
            
            # Calculate average response by numerosity (across all stimulus types)
            avg_by_num = []
            for num in unique_nums:
                num_mask = numerosities == num
                avg_by_num.append(np.mean(unit_acts[num_mask]))
            
            # Normalize to [0,1] for better comparison if values span different ranges
            avg_by_num = np.array(avg_by_num)
            if np.max(avg_by_num) - np.min(avg_by_num) > 1e-10:
                avg_by_num = (avg_by_num - np.min(avg_by_num)) / (np.max(avg_by_num) - np.min(avg_by_num))
            
            plt.plot(unique_nums, avg_by_num, marker='o', linewidth=1.5, label=f'Unit {unit_idx}')
        
        plt.xlabel('Numerosity', fontsize=14)
        plt.ylabel('Normalized Activation', fontsize=14)
        plt.title(f'{layer_name} - Response to Numerosity (Sample Units)', fontsize=16)
        plt.grid(True, alpha=0.3)
        # Only show legend for small number of units
        if len(selected_units) <= 10:
            plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{layer_name}_numerosity_response.png'))
        plt.close()
        
        # Also visualize on log scale to check for Weber-Fechner law pattern
        plt.figure(figsize=(15, 10))
        for unit_idx in selected_units:
            unit_acts = layer_acts_flat[:, unit_idx]
            
            # Calculate average response by numerosity (across all stimulus types)
            avg_by_num = []
            for num in unique_nums:
                num_mask = numerosities == num
                avg_by_num.append(np.mean(unit_acts[num_mask]))
            
            # Normalize to [0,1] for better comparison
            avg_by_num = np.array(avg_by_num)
            if np.max(avg_by_num) - np.min(avg_by_num) > 1e-10:
                avg_by_num = (avg_by_num - np.min(avg_by_num)) / (np.max(avg_by_num) - np.min(avg_by_num))
            
            plt.plot(np.log2(unique_nums), avg_by_num, marker='o', linewidth=1.5, label=f'Unit {unit_idx}')
        
        plt.xlabel('log2(Numerosity)', fontsize=14)
        plt.ylabel('Normalized Activation', fontsize=14)
        plt.title(f'{layer_name} - Response to Numerosity (Log Scale)', fontsize=16)
        plt.grid(True, alpha=0.3)
        if len(selected_units) <= 10:
            plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{layer_name}_numerosity_response_log.png'))
        plt.close()
        
        # For more detailed analysis, create separate plots for each stimulus type
        for stim_type in unique_stims:
            plt.figure(figsize=(15, 10))
            stim_mask = stimulus_types == stim_type
            
            for unit_idx in selected_units:
                unit_acts = layer_acts_flat[:, unit_idx]
                
                # Calculate average response by numerosity for this stimulus type
                avg_by_num = []
                for num in unique_nums:
                    mask = (numerosities == num) & stim_mask
                    if np.any(mask):
                        avg_by_num.append(np.mean(unit_acts[mask]))
                    else:
                        avg_by_num.append(np.nan)
                
                # Normalize to [0,1] for better comparison
                avg_by_num = np.array(avg_by_num)
                valid_idx = ~np.isnan(avg_by_num)
                if valid_idx.any() and np.max(avg_by_num[valid_idx]) - np.min(avg_by_num[valid_idx]) > 1e-10:
                    min_val = np.min(avg_by_num[valid_idx])
                    max_val = np.max(avg_by_num[valid_idx])
                    avg_by_num[valid_idx] = (avg_by_num[valid_idx] - min_val) / (max_val - min_val)
                
                plt.plot(unique_nums, avg_by_num, marker='o', linewidth=1.5, label=f'Unit {unit_idx}')
            
            plt.xlabel('Numerosity', fontsize=14)
            plt.ylabel('Normalized Activation', fontsize=14)
            plt.title(f'{layer_name} - Response to Numerosity ({stim_type})', fontsize=16)
            plt.grid(True, alpha=0.3)
            # Only show legend for small number of units
            if len(selected_units) <= 10:
                plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{layer_name}_{stim_type}_response.png'))
            plt.close()


def analyze_layer_activations(results, output_dir, model_name):
    """
    Perform basic statistical analysis on layer activations
    
    Args:
        results: Dictionary with activations and metadata
        output_dir: Directory to save results
        model_name: Name of the model (for subdirectory)
    """
    # Create analysis directory for this model
    analysis_dir = os.path.join(output_dir, model_name, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Get activations and metadata
    activations = results['activations']
    numerosities = results['numerosities']
    stimulus_types = results['stimulus_types']
    
    # Get unique numerosities and stimulus types
    unique_nums = np.sort(np.unique(numerosities))
    
    # For each layer, analyze activation patterns
    layer_stats = {}
    for layer_name, layer_acts in activations.items():
        print(f"Analyzing layer: {layer_name} with shape {layer_acts.shape}")
        
        # Properly flatten convolutional layers to preserve all neurons
        if layer_acts.ndim > 2:
            # For CNN layers, reshape to [batch_size, channels*height*width]
            batch_size = layer_acts.shape[0]
            layer_acts_flat = layer_acts.reshape(batch_size, -1)
            print(f"  Reshaped from {layer_acts.shape} to {layer_acts_flat.shape}")
        else:
            layer_acts_flat = layer_acts
        
        # Calculate basic statistics
        n_units = layer_acts_flat.shape[1]
        print(f"  Analyzing {n_units} total neurons")
        
        # Calculate average activation by numerosity for each unit
        unit_responses = np.zeros((n_units, len(unique_nums)))
        
        for i, num in enumerate(unique_nums):
            mask = numerosities == num
            unit_responses[:, i] = np.mean(layer_acts_flat[mask], axis=0)
        
        # Calculate metrics for each unit
        max_responses = np.max(unit_responses, axis=1)
        min_responses = np.min(unit_responses, axis=1)
        mean_responses = np.mean(unit_responses, axis=1)
        std_responses = np.std(unit_responses, axis=1)
        
        # Calculate preferred numerosity for each unit
        preferred_nums = unique_nums[np.argmax(unit_responses, axis=1)]
        
        # Calculate activation range relative to the mean
        # (max-min)/mean - high values suggest more selective units
        selectivity_index = (max_responses - min_responses) / (mean_responses + 1e-10)
        
        # Save unit stats
        unit_stats = pd.DataFrame({
            'unit_idx': np.arange(n_units),
            'max_response': max_responses,
            'min_response': min_responses,
            'mean_response': mean_responses, 
            'std_response': std_responses,
            'preferred_numerosity': preferred_nums,
            'selectivity_index': selectivity_index
        })
        
        # Sort by selectivity index
        unit_stats = unit_stats.sort_values('selectivity_index', ascending=False)
        
        # Save top N most selective units
        top_n = min(1000, n_units)  # Limit to 1000 units for large layers
        unit_stats.head(top_n).to_csv(os.path.join(analysis_dir, f"{layer_name}_top{top_n}_unit_stats.csv"), index=False)
        
        # For very large layers, save a random sample for inspection
        if n_units > 10000:
            sample_idx = np.random.choice(n_units, 5000, replace=False)
            sampled_stats = unit_stats.iloc[sample_idx]
            sampled_stats.to_csv(os.path.join(analysis_dir, f"{layer_name}_sampled_unit_stats.csv"), index=False)
        else:
            # Save complete stats for smaller layers
            unit_stats.to_csv(os.path.join(analysis_dir, f"{layer_name}_unit_stats.csv"), index=False)
        
        # Calculate original layer shape information for reference
        original_shape = "x".join(str(dim) for dim in layer_acts.shape[1:])
        
        # Store summary statistics
        layer_stats[layer_name] = {
            'original_shape': original_shape,
            'n_units': n_units,
            'mean_selectivity': float(np.mean(selectivity_index)),
            'median_selectivity': float(np.median(selectivity_index)),
            'top10_selectivity': float(np.mean(np.sort(selectivity_index)[-n_units//10:])),
            'top10_units': [int(idx) for idx in unit_stats.head(10)['unit_idx'].values],
            'preferred_num_counts': {int(num): int(np.sum(preferred_nums == num)) for num in unique_nums}
        }
        
        # Create distribution of preferred numerosities
        plt.figure(figsize=(12, 8))
        counts = np.bincount(np.searchsorted(unique_nums, preferred_nums), minlength=len(unique_nums))
        plt.bar(unique_nums, counts)
        plt.xlabel('Preferred Numerosity', fontsize=14)
        plt.ylabel('Number of Units', fontsize=14)
        plt.title(f'{layer_name} - Distribution of Preferred Numerosities', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f'{layer_name}_preferred_num_dist.png'))
        plt.close()
        
        # Plot selectivity index distribution
        plt.figure(figsize=(12, 8))
        plt.hist(selectivity_index, bins=30)
        plt.xlabel('Selectivity Index', fontsize=14)
        plt.ylabel('Number of Units', fontsize=14)
        plt.title(f'{layer_name} - Distribution of Selectivity Indices', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f'{layer_name}_selectivity_dist.png'))
        plt.close()
        
        # For the top 20 most selective units, plot their response curves
        plt.figure(figsize=(15, 10))
        top_units = unit_stats.head(20)['unit_idx'].values
        
        for unit_idx in top_units:
            response = unit_responses[unit_idx]
            # Normalize to [0,1] for better comparison
            response = (response - np.min(response)) / (np.max(response) - np.min(response) + 1e-10)
            plt.plot(unique_nums, response, marker='o', linewidth=1.5, 
                     label=f'Unit {unit_idx} (SI={selectivity_index[unit_idx]:.2f})')
        
        plt.xlabel('Numerosity', fontsize=14)
        plt.ylabel('Normalized Response', fontsize=14)
        plt.title(f'{layer_name} - Top Selective Units Response to Numerosity', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f'{layer_name}_top_units_response.png'))
        plt.close()
    
    # Save layer stats summary
    with open(os.path.join(analysis_dir, 'layer_stats_summary.json'), 'w') as f:
        json.dump(layer_stats, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Test neural network responses to numerosity stimuli")
    parser.add_argument('--data_dir', type=str, default='/home/embody_data/Numerosity_Emergy_CNN/numerosity_datasets',
                        help='Directory containing numerosity datasets')
    parser.add_argument('--output_dir', type=str, default='activation_data',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--csv_file', type=str, default='/home/embody_data/Numerosity_Emergy_CNN/numerosity_datasets/test.csv',
                        help='CSV file with image paths and metadata (default: look for test.csv or labels.csv in data_dir)')
    parser.add_argument('--model', type=str, 
                        choices=['CNN_1', 'CNN_2', 'CNN_3', 'CNN_4', 'CNN_5',
                                'MLP_1', 'MLP_2', 'MLP_3', 'MLP_4', 'MLP_5'], 
                        default='CNN_1',
                        help='Which model to test')
    parser.add_argument('--seed', type=int, default=21,
                        help='Random seed for reproducibility')
    parser.add_argument('--init_method', type=str, 
                        choices=['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 
                                'kaiming_normal', 'orthogonal', 'normal', 'uniform'],
                        default='xavier_uniform',
                        help='Weight initialization method')
    parser.add_argument('--save_incremental', action='store_true',
                        help='Save activations incrementally to save memory')
    parser.add_argument('--incremental_size', type=int, default=10,
                        help='Number of batches to process before saving incrementally')
    parser.add_argument('--cuda_deterministic', action='store_true',
                        help='Make CUDA operations deterministic')
    parser.add_argument('--save_weights', action='store_true',
                        help='Save model weights for reproducibility')
    parser.add_argument('--skip_vis', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--basic_analysis', action='store_true',
                        help='Perform basic statistical analysis on activations')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_random_seeds(args.seed, cuda_deterministic=args.cuda_deterministic)
    
    # Create experiment ID based on parameters
    experiment_id = f"{args.model}_seed{args.seed}_{args.init_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directory
    experiment_dir = os.path.join(args.output_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment configuration
    with open(os.path.join(experiment_dir, 'experiment_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Determine device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine CSV file if not specified
    if args.csv_file is None:
        # Try test.csv first, then fall back to labels.csv
        test_csv = os.path.join(args.data_dir, 'test.csv')
        labels_csv = os.path.join(args.data_dir, 'labels.csv')
        
        if os.path.exists(test_csv):
            args.csv_file = test_csv
        elif os.path.exists(labels_csv):
            args.csv_file = labels_csv
        else:
            raise FileNotFoundError(f"No CSV file found in {args.data_dir}")
    
    print(f"Using CSV file: {args.csv_file}")
    
    # Create dataset and dataloader
    dataset = NumerosityDataset(
        csv_file=args.csv_file,
        root_dir=args.data_dir
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Dataset loaded: {len(dataset)} images")
    
    # Initialize the model based on the selection
    if args.model == 'CNN_1':
        model = SingleLayerCNN()
    elif args.model == 'CNN_2':
        model = TwoLayerCNN()
    elif args.model == 'CNN_3':
        model = ThreeLayerCNN()
    elif args.model == 'CNN_4':
        model = FourLayerCNN()
    elif args.model == 'CNN_5':
        model = AlexNetCNN()
    elif args.model == 'MLP_1':
        model = SingleLayerMLP()
    elif args.model == 'MLP_2':
        model = TwoLayerMLP()
    elif args.model == 'MLP_3':
        model = ThreeLayerMLP()
    elif args.model == 'MLP_4':
        model = FourLayerMLP()
    elif args.model == 'MLP_5':
        model = FiveLayerMLP()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Initialize weights with the specified method
    init_weights(model, init_method=args.init_method)
    
    # Move model to device
    model = model.to(device)
    
    # Save model weights before activation extraction if requested
    if args.save_weights:
        save_model_weights(model, experiment_dir, args.model, 
                          init_method=args.init_method, seed=args.seed)
    
    # Extract activations
    if args.save_incremental:
        # Extract and save incrementally
        incremental_dir = os.path.join(experiment_dir, "incremental")
        results = extract_activations(
            model, dataloader, device, 
            save_dir=incremental_dir, 
            layer_batch_size=args.incremental_size
        )
    else:
        # Extract and save all at once
        results = extract_activations(model, dataloader, device)
        save_activations(results, experiment_dir, args.model)
    
    # Generate visualizations if not skipped
    if not args.skip_vis:
        create_basic_visualizations(results, experiment_dir, args.model)
    
    # Perform basic statistical analysis if requested
    if args.basic_analysis:
        analyze_layer_activations(results, experiment_dir, args.model)
    
    print(f"\nAll done! Model responses have been saved to {experiment_dir}")


if __name__ == '__main__':
    main()