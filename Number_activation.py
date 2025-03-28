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

# Import the models
from Models import SingleLayerCNN, TwoLayerCNN, ThreeLayerCNN, FourLayerCNN, AlexNetCNN, SingleLayerMLP, TwoLayerMLP, ThreeLayerMLP, FourLayerMLP, FiveLayerMLP


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


def extract_activations(model, dataloader, device):
    """
    Extract activations from all layers of the model for the given stimuli
    
    Args:
        model: Neural network model
        dataloader: DataLoader with numerosity stimuli
        device: Device to use for computation
    
    Returns:
        Dictionary containing activations and metadata
    """
    # Storage for activations and metadata
    all_activations = defaultdict(list)
    all_numerosities = []
    all_stimulus_types = []
    
    # Set model to evaluation mode
    model.eval()
    
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
    
    # Concatenate batch results
    for layer_name in all_activations:
        all_activations[layer_name] = np.concatenate(all_activations[layer_name], axis=0)
    
    # Convert lists to arrays
    all_numerosities = np.array(all_numerosities)
    all_stimulus_types = np.array(all_stimulus_types)
    
    return {
        'activations': all_activations,
        'numerosities': all_numerosities,
        'stimulus_types': all_stimulus_types
    }


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
            stimulus_types=stimulus_types
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
        
        # For convolutional layers, average across spatial dimensions
        if layer_acts.ndim > 2:
            layer_acts_flat = np.mean(layer_acts, axis=(2, 3))
        else:
            layer_acts_flat = layer_acts
        
        # Select a subset of units to visualize (randomly)
        n_units = layer_acts_flat.shape[1]
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
            
            plt.plot(unique_nums, avg_by_num, marker='o', linewidth=1.5, label=f'Unit {unit_idx}')
        
        plt.xlabel('Numerosity', fontsize=14)
        plt.ylabel('Average Activation', fontsize=14)
        plt.title(f'{layer_name} - Response to Numerosity (Sample Units)', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{layer_name}_numerosity_response.png'))
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
                
                plt.plot(unique_nums, avg_by_num, marker='o', linewidth=1.5, label=f'Unit {unit_idx}')
            
            plt.xlabel('Numerosity', fontsize=14)
            plt.ylabel('Average Activation', fontsize=14)
            plt.title(f'{layer_name} - Response to Numerosity ({stim_type})', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{layer_name}_{stim_type}_response.png'))
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test CNN responses to numerosity stimuli")
    parser.add_argument('--data_dir', type=str, default='numerosity_datasets',
                        help='Directory containing numerosity datasets')
    parser.add_argument('--output_dir', type=str, default='activation_data',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--csv_file', type=str, default=None,
                        help='CSV file with image paths and metadata (default: look for test.csv or labels.csv in data_dir)')
    parser.add_argument('--model', type=str, choices=['CNN_1', 'CNN_2', 'CNN_3','CNN_4','CNN_5',
                                                      'MLP_1','MLP_2','MLP_3','MLP_4','MLP_5'], default='CNN_1',
                        help='Which model to test')
    
    args = parser.parse_args()
    
    # Determine device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Initialize and test models
    
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

    model = model.to(device)
    
    results = extract_activations(model, dataloader, device)
    save_activations(results, args.output_dir, str(args.model))
    create_basic_visualizations(results, args.output_dir, str(args.model))    
    print("\nAll done! Model responses to numerosity stimuli have been saved.")


if __name__ == '__main__':
    main()
