import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import random_split
from torch import nn

import argparse
import random
import numpy as np
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt
from setproctitle import setproctitle
import os
import optuna
import time

from models.visiontransformer.vit import VisionTransformer

from datasets.dataset import CustomImageDataset, custom_collate_fn

# setting seed
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(opt, trial=None):
    print(opt)
    # Set the random seed for reproducibility
    seed_everything(opt.seed)

    # Set up the device (GPU or CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters setup
    epochs = opt.epochs
    patience = opt.patience
    learning_rate = opt.learning_rate
    batch_size = opt.batch_size
    output_dir = opt.output_dir

    print('calculating mean and std...')
    sys.stdout.flush()

    # Load the dataset
    temp_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    all_dataset = CustomImageDataset(directory='/data/furuya/cifar/cifar100/train/', transform=temp_transform)

    # Obtain a DataLoader for calculating dataset mean and standard deviation
    temp_loader = DataLoader(all_dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    # Calculate mean and standard deviation from the dataset
    data = next(iter(temp_loader))[0]
    mean = data.mean([0, 2, 3])
    std = data.std([0, 2, 3])

    # Define transformations including normalization
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    print('loading dataset...')
    sys.stdout.flush()

    # Define transformations including normalization
    all_dataset = CustomImageDataset(directory='/data/furuya/cifar/cifar100/train/', transform=transform)

    # Set sizes for training and validation datasets
    train_size = int(0.8 * len(all_dataset))
    val_size = len(all_dataset) - train_size

    # Split dataset into training and validation datasets
    train_dataset, val_dataset = random_split(all_dataset, [train_size, val_size])

    # Obtain DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    # Model definition
    num_classes = 100

    # Model selection based on the option provided
    model = VisionTransformer(
        image_size=128,
        patch_size=16,
        in_channels=1,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=num_classes,
        hidden_dims=[64, 128, 256]
    )

    # Parallelize model if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Transfer model to the device
    model.to(device)

    model_state_dict = torch.load(opt.model_path)
    model.load_state_dict(model_state_dict)

    model = model.load_from_pretrained()

    params_to_update = [param for param in model.parameters() if param.requires_grad]

    # Initialize the optimizer
    optimizer = torch.optim.SGD(params_to_update, lr=learning_rate)

    # Initialize parameters for early stopping
    val_loss_min = None
    val_loss_min_epoch = 0

    # Initialize arrays for plotting the learning curve
    train_losses = []
    val_losses = []

    # Start model training
    for epoch in tqdm(range(epochs)):

        # Initialize parameters for training
        train_loss = 0.0
        val_loss = 0.0

        # Set model to training mode
        model.train()

        # Load training data
        for inputs, labels in train_loader:
            
            inputs, labels = inputs.to(device), labels.to(device)

            # Initialize gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Aggregate training loss
            train_loss += loss.item()

        # Calculate average training loss
        train_loss /= len(train_loader)
        # Record training loss
        train_losses.append(train_loss)

        # Set model to evaluation mode
        model.eval()

        val_correct = 0  # Correct predictions count
        val_total = 0    # Total predictions count
        
        # Evaluate on validation data
        with torch.no_grad():
            for inputs, labels in val_loader:
                
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1) 
                val_total += labels.size(0)      
                val_correct += (predicted == labels).sum().item() 

        val_loss /= len(val_loader)  # Calculate average validation loss
        val_accuracy = val_correct / val_total  # Calculate validation accuracy
        val_losses.append(val_loss)  # Record validation loss

        # Print epoch summary
        print(f'Epoch {epoch+1}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        sys.stdout.flush()

        # Optimize memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save model if validation loss improved
        if val_loss_min is None or val_loss < val_loss_min:
            model_save_directory = output_dir
            model_save_name = f'{output_dir}/lr{learning_rate}_ep{epochs}_pa{patience}intweak.pt'
            if not os.path.exists(model_save_directory):
                os.makedirs(model_save_directory)
            torch.save(model.state_dict(), model_save_name)
            val_loss_min = val_loss
            val_loss_min_epoch = epoch
            
        # Early stopping if no improvement in validation loss for a set number of epochs
        elif (epoch - val_loss_min_epoch) >= patience:
            print('Early stopping due to validation loss not improving for {} epochs'.format(patience))
            break

    # Plot and save the learning curve
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.title("Training and Validation Loss")
    graph_save_directory = output_dir
    graph_save_name = f'{graph_save_directory}/lr{learning_rate}_ep{epochs}_pa{patience}intweak.png'

    if not os.path.exists(graph_save_directory):
        os.makedirs(graph_save_directory)

    plt.savefig(graph_save_name)


    return val_loss_min

def objective(trial):
    args = argparse.Namespace(
        epochs=100,
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        patience=20,
        seed=42,
        batch_size=64,
        output_dir='./AddViT',
        model_path=''
    )
    return train(args, trial)

if __name__ == '__main__':
    # setting the name of the process
    setproctitle("addvittrain")

    print('-----biginning training-----')
    start_time = time.time()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    end_time = time.time()
    # calculate elapsed time
    elapsed_time = end_time - start_time
    print(f'-----completing training in {elapsed_time:.2f} seconds-----')