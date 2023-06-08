import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.models import vgg16

from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

import os
import numpy as np

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
        logging.FileHandler(f'log_files/logfile_vgg16_{cur_time}.log'),  # Specify the path to the log file
    ]
)

# Create a logger instance
logger = logging.getLogger()

def vgg16_attack_pregen_training():

    # Define the Resmodel model
    load_ep_num = 0
    # model = torch.load(f'saved_model/vgg16_pdg_training/vgg16_ep_{load_ep_num}_pdg_trained.pth')
    premodel = torch.load(f'pytorch/saved_model/vgg16_best_model.pth')
    model = torch.load(f'saved_model/vgg16_attack_pregen_training/vgg16_attack_pregen_best_model_2.pth')

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
    print("device: ", device)
    logger.info("device:")
    logger.info(device.type)
    premodel = premodel.to(device)
    model = model.to(device)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    best_loss = np.inf
    early_stop_counter = 0
    best_model_weights = None
    patience = 3
    num_epochs = 100
    save_model_dir = "saved_model/vgg16_attack_pregen_training_final/"
    os.makedirs(save_model_dir, exist_ok=True)
    train_adv_dataset=[]
    test_adv_dataset=[]

    batch = 0
    for images, labels in train_loader:
        print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
        print(f'batch: {batch+1}/{len(train_loader)}')
        images = images.to(device)[:25]
        labels = labels.to(device)[:25]

        adv_images = projected_gradient_descent(premodel, images, eps=0.1, eps_iter=0.01, nb_iter=10, norm=np.inf)
        train_adv_dataset.append([adv_images, labels])
        batch += 1

        if batch == 100:
            break

    batch = 0
    for images, labels in test_loader:
        print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
        print(f'batch: {batch+1}/{len(test_loader)}')
        images = images.to(device)[:25]
        labels = labels.to(device)[:25]

        adv_images = projected_gradient_descent(premodel, images, eps=0.1, eps_iter=0.01, nb_iter=10, norm=np.inf)
        test_adv_dataset.append([adv_images, labels])
        batch += 1

        # if batch == 10:
        #     break

    for epoch in range(num_epochs):
        print(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
        print(f"Training - epoch - {epoch+1}/{num_epochs}")
        logger.info(f"Training - epoch - {epoch+1}/{num_epochs}")
        model.train()  # Set the model to training mode
        train_loss = 0
        correct = 0
        total = 0

        batch = 0
        for images, labels in train_loader:
            print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
            print(f'batch: {batch+1}/{len(train_loader)}')
            images = images.to(device)
            labels = labels.to(device)
            adv_images, adv_labels = train_adv_dataset[batch]
            images = torch.cat([images, adv_images], dim=0)
            labels = torch.cat([labels, adv_labels], dim=0)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch += 1

            if batch == 100:
                break

        train_accuracy = correct / total

       # Evaluate on the test set
        model.eval()  # Set the model to evaluation mode
        orig_correct = 0
        adv_correct = 0
        orig_total = 0
        adv_total = 0
        total = 0
        orig_val_loss = 0
        adv_val_loss = 0
        tot_val_loss = 0
        batch = 0
        print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
        for images, labels in test_loader:
            # print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
            # print(f'batch: {batch+1}/{len(test_loader)}')

            # eval original
            images = images.to(device)
            labels = labels.to(device)
            orig_total += labels.size(0)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            orig_val_loss = loss_fn(outputs, labels)
            tot_val_loss += orig_val_loss.item() * images.size(0)
            orig_val_loss += orig_val_loss.item() * images.size(0)
            orig_correct += (predicted == labels).sum().item()

            # eval adv
        for images, labels in test_adv_dataset:
            adv_images = images.to(device)
            adv_labels = labels.to(device)
            adv_total += adv_labels.size(0)
            adv_outputs = model(adv_images)
            _, adv_predicted = torch.max(adv_outputs.data, 1)
            adv_val_loss = loss_fn(adv_outputs, adv_labels)
            tot_val_loss += adv_val_loss.item() * adv_images.size(0)
            adv_val_loss += adv_val_loss.item() * adv_images.size(0)
            adv_correct += (adv_predicted == adv_labels).sum().item()

            batch += 1
            # break

        orig_val_accuracy = orig_correct / orig_total
        adv_val_accuracy = adv_correct / adv_total
        tot_val_accuracy = (orig_correct + adv_correct) / (orig_total + adv_total) 

        torch.save(model, f'{save_model_dir}/vgg16_ep_{epoch+1}_attack_pregen_trained.pth')
        print(f"Saved {save_model_dir}/vgg16_ep_{epoch+1}_attack_pregen_trained.pth")
        print(f"Epoch [{epoch+1}/{num_epochs}], train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, orig_val_loss: {orig_val_loss:.4f}, orig_val_accuracy: {orig_val_accuracy:.4f}, adv_val_loss: {adv_val_loss:.4f}, adv_val_accuracy: {adv_val_accuracy:.4f}, tot_val_loss: {tot_val_loss:.4f}, tot_val_accuracy: {tot_val_accuracy:.4f}")
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, orig_val_loss: {orig_val_loss:.4f}, orig_val_accuracy: {orig_val_accuracy:.4f}, adv_val_loss: {adv_val_loss:.4f}, adv_val_accuracy: {adv_val_accuracy:.4f}, tot_val_loss: {tot_val_loss:.4f}, tot_val_accuracy: {tot_val_accuracy:.4f}")

        if tot_val_loss < best_loss:
            best_loss = tot_val_loss
            early_stop_counter = 0
            # Save the best model weights
            best_model_weights = model.state_dict()
        else:
            early_stop_counter += 1

        if epoch >= 80 and early_stop_counter >= patience:
            print(f"Early stopping at {epoch+1} after {patience} epochs without improvement.")
            break
    
    model.load_state_dict(best_model_weights)
    torch.save(model, f'{save_model_dir}/vgg16_attack_pregen_best_model.pth')
    print(f"Saved {save_model_dir}/vgg16_attack_pregen_best_model.pth")



def model_evaluation():
    # premodel = torch.load(f'saved_model/vgg16_attack_pregen_training/vgg16_attack_pregen_best_model_2.pth')
    premodel = torch.load(f'/mnt/task_runtime/pytorch/saved_model/vgg16_best_model.pth')
    model = torch.load(f'saved_model/vgg16_attack_pregen_training/vgg16_attack_pregen_best_model.pth')

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
    print("device: ", device)
    logger.info("device:")
    logger.info(device.type)
    premodel = premodel.to(device)
    model = model.to(device)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()

    test_adv_dataset=[]

    batch = 0
    for images, labels in test_loader:
        print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
        print(f'batch: {batch+1}/{len(test_loader)}')
        images = images.to(device)[:25]
        labels = labels.to(device)[:25]

        adv_images = projected_gradient_descent(premodel, images, eps=0.1, eps_iter=0.01, nb_iter=10, norm=np.inf)
        test_adv_dataset.append([adv_images, labels])
        batch += 1


    # Evaluate on the test set
    model.eval()  # Set the model to evaluation mode
    orig_correct = 0
    adv_correct = 0
    orig_total = 0
    adv_total = 0
    total = 0
    orig_val_loss = 0
    adv_val_loss = 0
    tot_val_loss = 0
    batch = 0
    print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
    for images, labels in test_loader:
        # print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
        # print(f'batch: {batch+1}/{len(test_loader)}')

        # eval original
        images = images.to(device)
        labels = labels.to(device)
        orig_total += labels.size(0)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        orig_val_loss = loss_fn(outputs, labels)
        tot_val_loss += orig_val_loss.item() * images.size(0)
        orig_val_loss += orig_val_loss.item() * images.size(0)
        orig_correct += (predicted == labels).sum().item()

        # eval adv
    for images, labels in test_adv_dataset:
        adv_images = images.to(device)
        adv_labels = labels.to(device)
        adv_total += adv_labels.size(0)
        adv_outputs = model(adv_images)
        _, adv_predicted = torch.max(adv_outputs.data, 1)
        adv_val_loss = loss_fn(adv_outputs, adv_labels)
        tot_val_loss += adv_val_loss.item() * adv_images.size(0)
        adv_val_loss += adv_val_loss.item() * adv_images.size(0)
        adv_correct += (adv_predicted == adv_labels).sum().item()

        batch += 1
        # break

    orig_val_accuracy = orig_correct / orig_total
    adv_val_accuracy = adv_correct / adv_total
    tot_val_accuracy = (orig_correct + adv_correct) / (orig_total + adv_total)

    print(f"orig_val_loss: {orig_val_loss:.4f}, orig_val_accuracy: {orig_val_accuracy:.4f}, adv_val_loss: {adv_val_loss:.4f}, adv_val_accuracy: {adv_val_accuracy:.4f}, tot_val_loss: {tot_val_loss:.4f}, tot_val_accuracy: {tot_val_accuracy:.4f}")
    logger.info(f"orig_val_loss: {orig_val_loss:.4f}, orig_val_accuracy: {orig_val_accuracy:.4f}, adv_val_loss: {adv_val_loss:.4f}, adv_val_accuracy: {adv_val_accuracy:.4f}, tot_val_loss: {tot_val_loss:.4f}, tot_val_accuracy: {tot_val_accuracy:.4f}")

# model_evaluation()




vgg16_attack_pregen_training()