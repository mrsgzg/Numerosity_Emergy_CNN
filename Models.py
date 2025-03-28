import torch
import torch.nn as nn
import torch.nn.functional as F
import math

###################
# CNN MODELS
###################

class SingleLayerCNN(nn.Module):
    """
    Simple 1-layer CNN for numerosity detection
    """
    
    def __init__(self, num_classes=50):
        super(SingleLayerCNN, self).__init__()
        
        # Single convolutional layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x, return_features=False):
        # Convolutional layer
        x1 = F.relu(self.bn1(self.conv1(x)))
        block1_features = x1
        x = self.pool1(x1)
        
        # Global average pooling
        x = self.gap(x)
        pooled_features = x
        
        # Flatten
        x_flat = torch.flatten(x, 1)
        
        # Fully connected layer
        output = self.fc(x_flat)
        
        if return_features:
            return {
                'block1': block1_features,
                'pooled': pooled_features,
                'flattened': x_flat,
                'output': output
            }
        
        return output
    
    def get_activation_maps(self, x, layer_name):
        """Get activation maps for visualization"""
        if layer_name == 'block1':
            return F.relu(self.bn1(self.conv1(x)))
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")


class TwoLayerCNN(nn.Module):
    """
    2-layer CNN for numerosity detection
    """
    
    def __init__(self, num_classes=50):
        super(TwoLayerCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x, return_features=False):
        # Block 1
        x1 = F.relu(self.bn1(self.conv1(x)))
        block1_features = x1
        x = self.pool1(x1)
        
        # Block 2
        x2 = F.relu(self.bn2(self.conv2(x)))
        block2_features = x2
        x = self.pool2(x2)
        
        # Global average pooling
        x = self.gap(x)
        pooled_features = x
        
        # Flatten
        x_flat = torch.flatten(x, 1)
        
        # Fully connected layer
        output = self.fc(x_flat)
        
        if return_features:
            return {
                'block1': block1_features,
                'block2': block2_features,
                'pooled': pooled_features,
                'flattened': x_flat,
                'output': output
            }
        
        return output
    
    def get_activation_maps(self, x, layer_name):
        """Get activation maps for visualization"""
        if layer_name == 'block1':
            return F.relu(self.bn1(self.conv1(x)))
        elif layer_name == 'block2':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            return F.relu(self.bn2(self.conv2(x)))
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")


class ThreeLayerCNN(nn.Module):
    """
    3-layer CNN for numerosity detection
    """
    
    def __init__(self, num_classes=50):
        super(ThreeLayerCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x, return_features=False):
        # Block 1
        x1 = F.relu(self.bn1(self.conv1(x)))
        block1_features = x1
        x = self.pool1(x1)
        
        # Block 2
        x2 = F.relu(self.bn2(self.conv2(x)))
        block2_features = x2
        x = self.pool2(x2)
        
        # Block 3
        x3 = F.relu(self.bn3(self.conv3(x)))
        block3_features = x3
        x = self.pool3(x3)
        
        # Global average pooling
        x = self.gap(x)
        pooled_features = x
        
        # Flatten
        x_flat = torch.flatten(x, 1)
        
        # Fully connected layer
        output = self.fc(x_flat)
        
        if return_features:
            return {
                'block1': block1_features,
                'block2': block2_features,
                'block3': block3_features,
                'pooled': pooled_features,
                'flattened': x_flat,
                'output': output
            }
        
        return output
    
    def get_activation_maps(self, x, layer_name):
        """Get activation maps for visualization"""
        if layer_name == 'block1':
            return F.relu(self.bn1(self.conv1(x)))
        elif layer_name == 'block2':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            return F.relu(self.bn2(self.conv2(x)))
        elif layer_name == 'block3':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            return F.relu(self.bn3(self.conv3(x)))
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")


class FourLayerCNN(nn.Module):
    """
    4-layer CNN for numerosity detection
    """
    
    def __init__(self, num_classes=50):
        super(FourLayerCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x, return_features=False):
        # Block 1
        x1 = F.relu(self.bn1(self.conv1(x)))
        block1_features = x1
        x = self.pool1(x1)
        
        # Block 2
        x2 = F.relu(self.bn2(self.conv2(x)))
        block2_features = x2
        x = self.pool2(x2)
        
        # Block 3
        x3 = F.relu(self.bn3(self.conv3(x)))
        block3_features = x3
        x = self.pool3(x3)
        
        # Block 4
        x4 = F.relu(self.bn4(self.conv4(x)))
        block4_features = x4
        x = self.pool4(x4)
        
        # Global average pooling
        x = self.gap(x)
        pooled_features = x
        
        # Flatten
        x_flat = torch.flatten(x, 1)
        
        # Fully connected layer
        output = self.fc(x_flat)
        
        if return_features:
            return {
                'block1': block1_features,
                'block2': block2_features,
                'block3': block3_features,
                'block4': block4_features,
                'pooled': pooled_features,
                'flattened': x_flat,
                'output': output
            }
        
        return output
    
    def get_activation_maps(self, x, layer_name):
        """Get activation maps for visualization"""
        if layer_name == 'block1':
            return F.relu(self.bn1(self.conv1(x)))
        elif layer_name == 'block2':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            return F.relu(self.bn2(self.conv2(x)))
        elif layer_name == 'block3':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            return F.relu(self.bn3(self.conv3(x)))
        elif layer_name == 'block4':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            return F.relu(self.bn4(self.conv4(x)))
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")


class AlexNetCNN(nn.Module):
    """
    Modified AlexNet for numerosity detection
    """
    
    def __init__(self, num_classes=50):
        super(AlexNetCNN, self).__init__()
        
        # Features layers (modified AlexNet)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Adaptive pooling to ensure fixed size output regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier layers
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, return_features=False):
        # Feature extraction
        x1 = F.relu(self.bn1(self.conv1(x)))
        block1_features = x1
        x = self.pool1(x1)
        
        x2 = F.relu(self.bn2(self.conv2(x)))
        block2_features = x2
        x = self.pool2(x2)
        
        x3 = F.relu(self.bn3(self.conv3(x)))
        block3_features = x3
        
        x4 = F.relu(self.bn4(self.conv4(x3)))
        block4_features = x4
        
        x5 = F.relu(self.bn5(self.conv5(x4)))
        block5_features = x5
        x = self.pool5(x5)
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        
        # Flatten
        x_flat = torch.flatten(x, 1)
        
        # Fully connected layers
        fc1_features = F.relu(self.fc1(x_flat))
        fc1_dropout = self.dropout(fc1_features)
        
        fc2_features = F.relu(self.fc2(fc1_dropout))
        fc2_dropout = self.dropout(fc2_features)
        
        output = self.fc3(fc2_dropout)
        
        if return_features:
            return {
                'block1': block1_features,
                'block2': block2_features,
                'block3': block3_features,
                'block4': block4_features,
                'block5': block5_features,
                'flattened': x_flat,
                'fc1': fc1_features,
                'fc2': fc2_features,
                'output': output
            }
        
        return output
    
    def get_activation_maps(self, x, layer_name):
        """Get activation maps for visualization"""
        if layer_name == 'block1':
            return F.relu(self.bn1(self.conv1(x)))
        elif layer_name == 'block2':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            return F.relu(self.bn2(self.conv2(x)))
        elif layer_name == 'block3':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            return F.relu(self.bn3(self.conv3(x)))
        elif layer_name == 'block4':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            return F.relu(self.bn4(self.conv4(x)))
        elif layer_name == 'block5':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            return F.relu(self.bn5(self.conv5(x)))
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")


###################
# MLP MODELS
###################

class SingleLayerMLP(nn.Module):
    """
    Single hidden layer MLP for numerosity detection
    """
    
    def __init__(self, num_classes=50, input_size=(1, 240, 320)):
        super(SingleLayerMLP, self).__init__()
        
        # Calculate flattened input size
        self.input_dim = input_size[0] * input_size[1] * input_size[2]
        
        # Hidden layers
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.dropout1 = nn.Dropout(0.5)
        
        # Output layer
        self.fc_out = nn.Linear(512, num_classes)
        
    def forward(self, x, return_features=False):
        # Flatten input
        x_flat = torch.flatten(x, 1)
        input_features = x_flat
        
        # Hidden layer
        fc1_output = F.relu(self.fc1(x_flat))
        hidden1_features = fc1_output
        fc1_dropout = self.dropout1(fc1_output)
        
        # Output layer
        output = self.fc_out(fc1_dropout)
        
        if return_features:
            return {
                'input': input_features,
                'hidden1': hidden1_features,
                'output': output
            }
        
        return output


class TwoLayerMLP(nn.Module):
    """
    Two hidden layer MLP for numerosity detection
    """
    
    def __init__(self, num_classes=50, input_size=(1, 240, 320)):
        super(TwoLayerMLP, self).__init__()
        
        # Calculate flattened input size
        self.input_dim = input_size[0] * input_size[1] * input_size[2]
        
        # Hidden layers
        self.fc1 = nn.Linear(self.input_dim, 1024)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        
        # Output layer
        self.fc_out = nn.Linear(512, num_classes)
        
    def forward(self, x, return_features=False):
        # Flatten input
        x_flat = torch.flatten(x, 1)
        input_features = x_flat
        
        # Hidden layer 1
        fc1_output = F.relu(self.fc1(x_flat))
        hidden1_features = fc1_output
        fc1_dropout = self.dropout1(fc1_output)
        
        # Hidden layer 2
        fc2_output = F.relu(self.fc2(fc1_dropout))
        hidden2_features = fc2_output
        fc2_dropout = self.dropout2(fc2_output)
        
        # Output layer
        output = self.fc_out(fc2_dropout)
        
        if return_features:
            return {
                'input': input_features,
                'hidden1': hidden1_features,
                'hidden2': hidden2_features,
                'output': output
            }
        
        return output


class ThreeLayerMLP(nn.Module):
    """
    Three hidden layer MLP for numerosity detection
    """
    
    def __init__(self, num_classes=50, input_size=(1, 240, 320)):
        super(ThreeLayerMLP, self).__init__()
        
        # Calculate flattened input size
        self.input_dim = input_size[0] * input_size[1] * input_size[2]
        
        # Hidden layers
        self.fc1 = nn.Linear(self.input_dim, 2048)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(0.5)
        
        # Output layer
        self.fc_out = nn.Linear(512, num_classes)
        
    def forward(self, x, return_features=False):
        # Flatten input
        x_flat = torch.flatten(x, 1)
        input_features = x_flat
        
        # Hidden layer 1
        fc1_output = F.relu(self.fc1(x_flat))
        hidden1_features = fc1_output
        fc1_dropout = self.dropout1(fc1_output)
        
        # Hidden layer 2
        fc2_output = F.relu(self.fc2(fc1_dropout))
        hidden2_features = fc2_output
        fc2_dropout = self.dropout2(fc2_output)
        
        # Hidden layer 3
        fc3_output = F.relu(self.fc3(fc2_dropout))
        hidden3_features = fc3_output
        fc3_dropout = self.dropout3(fc3_output)
        
        # Output layer
        output = self.fc_out(fc3_dropout)
        
        if return_features:
            return {
                'input': input_features,
                'hidden1': hidden1_features,
                'hidden2': hidden2_features,
                'hidden3': hidden3_features,
                'output': output
            }
        
        return output


class FourLayerMLP(nn.Module):
    """
    Four hidden layer MLP for numerosity detection
    """
    
    def __init__(self, num_classes=50, input_size=(1, 240, 320)):
        super(FourLayerMLP, self).__init__()
        
        # Calculate flattened input size
        self.input_dim = input_size[0] * input_size[1] * input_size[2]
        
        # Hidden layers
        self.fc1 = nn.Linear(self.input_dim, 2048)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(512, 256)
        self.dropout4 = nn.Dropout(0.5)
        
        # Output layer
        self.fc_out = nn.Linear(256, num_classes)
        
    def forward(self, x, return_features=False):
        # Flatten input
        x_flat = torch.flatten(x, 1)
        input_features = x_flat
        
        # Hidden layer 1
        fc1_output = F.relu(self.fc1(x_flat))
        hidden1_features = fc1_output
        fc1_dropout = self.dropout1(fc1_output)
        
        # Hidden layer 2
        fc2_output = F.relu(self.fc2(fc1_dropout))
        hidden2_features = fc2_output
        fc2_dropout = self.dropout2(fc2_output)
        
        # Hidden layer 3
        fc3_output = F.relu(self.fc3(fc2_dropout))
        hidden3_features = fc3_output
        fc3_dropout = self.dropout3(fc3_output)
        
        # Hidden layer 4
        fc4_output = F.relu(self.fc4(fc3_dropout))
        hidden4_features = fc4_output
        fc4_dropout = self.dropout4(fc4_output)
        
        # Output layer
        output = self.fc_out(fc4_dropout)
        
        if return_features:
            return {
                'input': input_features,
                'hidden1': hidden1_features,
                'hidden2': hidden2_features,
                'hidden3': hidden3_features,
                'hidden4': hidden4_features,
                'output': output
            }
        
        return output


class FiveLayerMLP(nn.Module):
    """
    Five hidden layer MLP for numerosity detection
    """
    
    def __init__(self, num_classes=50, input_size=(1, 240, 320)):
        super(FiveLayerMLP, self).__init__()
        
        # Calculate flattened input size
        self.input_dim = input_size[0] * input_size[1] * input_size[2]
        
        # Hidden layers
        self.fc1 = nn.Linear(self.input_dim, 4096)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(4096, 2048)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(2048, 1024)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(1024, 512)
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc5 = nn.Linear(512, 256)
        self.dropout5 = nn.Dropout(0.5)
        
        # Output layer
        self.fc_out = nn.Linear(256, num_classes)
        
    def forward(self, x, return_features=False):
        # Flatten input
        x_flat = torch.flatten(x, 1)
        input_features = x_flat
        
        # Hidden layer 1
        fc1_output = F.relu(self.fc1(x_flat))
        hidden1_features = fc1_output
        fc1_dropout = self.dropout1(fc1_output)
        
        # Hidden layer 2
        fc2_output = F.relu(self.fc2(fc1_dropout))
        hidden2_features = fc2_output
        fc2_dropout = self.dropout2(fc2_output)
        
        # Hidden layer 3
        fc3_output = F.relu(self.fc3(fc2_dropout))
        hidden3_features = fc3_output
        fc3_dropout = self.dropout3(fc3_output)
        
        # Hidden layer 4
        fc4_output = F.relu(self.fc4(fc3_dropout))
        hidden4_features = fc4_output
        fc4_dropout = self.dropout4(fc4_output)
        
        # Hidden layer 5
        fc5_output = F.relu(self.fc5(fc4_dropout))
        hidden5_features = fc5_output
        fc5_dropout = self.dropout5(fc5_output)
        
        # Output layer
        output = self.fc_out(fc5_dropout)
        
        if return_features:
            return {
                'input': input_features,
                'hidden1': hidden1_features,
                'hidden2': hidden2_features,
                'hidden3': hidden3_features,
                'hidden4': hidden4_features,
                'hidden5': hidden5_features,
                'output': output
            }
        
        return output