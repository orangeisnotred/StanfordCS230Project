import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

import os
import numpy as np
import random
import logging
import datetime

os.makedirs('log_files', exist_ok = True)
# Configure logging
cur_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'log_files/logfile_resnet50_ens_{cur_time}.log'),  # Specify the path to the log file
    ]
)

# Create a logger instance
logger = logging.getLogger()

def resnet50_ensemble_attack_training():
    # Define the Resmodel model
    vgg16_model = torch.load('saved_model/vgg16_best_model.pth')
    resnet50_model = torch.load('saved_model/resnet50_best_model.pth')
    resnet101_model = torch.load('saved_model/resnet101_best_model.pth')
    
    # Load and preprocess the MNIST dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Set the device to use (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")
    print("device: ", device)
    logger.info("device:")
    logger.info(device.type)
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     vgg16_model = DataParallel(vgg16_model)
    #     resnet50_model = DataParallel(resnet50_model)
    #     resnet101_model = DataParallel(vgg16_model,)
    # vgg16_model = vgg16_model.cuda()
    # resnet50_model = resnet50_model.cuda()
    # resnet101_model = resnet101_model.cuda()

    vgg16_model = vgg16_model.to(device)
    resnet50_model = resnet50_model.to(device)
    resnet101_model = resnet101_model.to(device)
    models = [resnet50_model, vgg16_model, resnet101_model]


    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet50_model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    best_loss = np.inf
    early_stop_counter = 0
    best_model_weights = None
    patience = 3
    num_epochs = 50
    save_model_dir = "saved_model/resnet50_ens_pdg_training/"
    os.makedirs(save_model_dir, exist_ok=True)
    for epoch in range(num_epochs):
        print(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
        print(f"Training - epoch - {epoch+1}/{num_epochs}")
        logger.info(f"Training - epoch - {epoch+1}/{num_epochs}")
        resnet50_model.train()  # Set the model to training mode
        train_loss = 0
        correct = 0
        total = 0

        batch = 0
        for images, labels in train_loader:
            print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
            print(f'batch: {batch+1}/{len(train_loader)}')
            # images = images.cuda()
            # labels = labels.cuda()
            images = images.to(device)
            labels = labels.to(device)
            rand_num = random.randint(0, 2)
            model = models[rand_num]
            adv_images = projected_gradient_descent(model, images, eps=0.3, eps_iter=0.01, nb_iter=40, norm=np.inf)
            images = torch.cat([images, adv_images], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            optimizer.zero_grad()
            outputs = resnet50_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch += 1

        train_accuracy = correct / total

        # Evaluate on the test set
        print(f"Eval - epoch - {epoch+1}/{num_epochs}")
        logger.info(f"Eval - epoch - {epoch+1}/{num_epochs}")
        resnet50_model.eval()  # Set the model to evaluation mode
        correct = 0
        adv_correct = 0
        total = 0
        val_loss = 0
        adv_val_loss = 0
        batch = 0
        for images, labels in test_loader:
            print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
            print(f'batch: {batch+1}/{len(test_loader)}')
            # images = images.cuda()
            # labels = labels.cuda()
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = resnet50_model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_loss = loss_fn(outputs, labels)
            val_loss += val_loss.item() * images.size(0)
            correct += (predicted == labels).sum().item()

            rand_num = random.randint(0, 2)
            model = models[rand_num]
            adv_images = projected_gradient_descent(model, images, eps=0.3, eps_iter=0.01, nb_iter=40, norm=np.inf)
            adv_outputs = resnet50_model(adv_images)
            _, adv_predicted = torch.max(adv_outputs.data, 1)
            adv_val_loss = loss_fn(adv_outputs, labels)
            adv_val_loss += adv_val_loss.item() * images.size(0)
            adv_correct += (adv_predicted == labels).sum().item()

            batch += 1

        val_accuracy = correct / total
        adv_val_accuracy = adv_correct / total

        torch.save(model, f'{save_model_dir}/resnet50_ep_{epoch+1}_ens_pdg_trained.pth')
        print(f"Saved {save_model_dir}/resnet50_ep_{epoch+1}_ens_pdg_trained.pth")
        print(f"Epoch [{epoch+1}/{num_epochs}], train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}, adv_val_loss: {adv_val_loss:.4f}, adv_val_accuracy: {adv_val_accuracy:.4f}")
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}, adv_val_loss: {adv_val_loss:.4f}, adv_val_accuracy: {adv_val_accuracy:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            # Save the best model weights
            best_model_weights = model.state_dict()
        else:
            early_stop_counter += 1

        if epoch >= 20 and early_stop_counter >= patience:
            print(f"Early stopping at {epoch+1} after {patience} epochs without improvement.")
            break
    
    model.load_state_dict(best_model_weights)
    torch.save(model, f'{save_model_dir}/resnet50_best_model_ens_pdg_trained.pth')
    print(f"Saved {save_model_dir}/resnet50_best_model_ens_pdg_trained.pth")
