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


def generate_adverasrial_examples(model, dataset, device=None, resize=None, early_stop=None):
    orig_dataset = []
    adv_dataset=[]
    if device:
        model = model.to(device)

    batch = 0
    for images, labels in dataset:
        print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
        print(f'batch: {batch+1}/{len(dataset)}')
        if device:
            images = images.to(device)[:25]
            labels = labels.to(device)[:25]

        adv_images = projected_gradient_descent(model, images, eps=0.1, eps_iter=0.01, nb_iter=10, norm=np.inf)
        if resize is not None:
            resize_transform = transforms.Resize(resize)
            adv_images = resize_transform(adv_images)
            images = resize_transform(images)

        adv_dataset.append([adv_images, labels])
        orig_dataset.append([images, labels])
        batch += 1

        if early_stop and batch == early_stop:
            break

    return orig_dataset, adv_dataset


def models_ensemble_attack_pregen_training():

    vgg16_premodel = torch.load(f'saved_model/vgg16_best_model.pth')
    vgg16_model = torch.load(f'/mnt/task_runtime/saved_model/models_ens_attack_pregen_training_final/vgg16_model_attack_pregen_best_model_final_2.pth')
    resnet50_premodel = torch.load(f'saved_model/resnet50_best_model.pth')
    resnet50_model = torch.load(f'/mnt/task_runtime/saved_model/models_ens_attack_pregen_training_final/resnet50_model_attack_pregen_best_model_final_2.pth')
    resnet101_premodel = torch.load(f'saved_model/resnet101_best_model.pth')
    resnet101_model = torch.load(f'/mnt/task_runtime/saved_model/models_ens_attack_pregen_training_final/resnet101_model_attack_pregen_best_model_final_2.pth')

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

    device = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:2")
    device3 = torch.device("cuda:3")
    device4 = torch.device("cuda:4")
    device5 = torch.device("cuda:5")
    device6 = torch.device("cuda:6")
    device7 = torch.device("cuda:7")

    vgg16_premodel = vgg16_premodel.to(device1)
    vgg16_model = vgg16_model.to(device1)
    resnet50_premodel = resnet50_premodel.to(device2)
    resnet50_model = resnet50_model.to(device2)
    resnet101_premodel = resnet101_premodel.to(device3)
    resnet101_model = resnet101_model.to(device3)

    _, train_vgg16_adv_dataset = generate_adverasrial_examples(vgg16_premodel, train_loader, device=device1, early_stop=100)
    _, test_vgg16_adv_dataset = generate_adverasrial_examples(vgg16_premodel, test_loader, device=device1)
    _, train_resnet50_adv_dataset = generate_adverasrial_examples(resnet50_premodel, train_loader, device=device2, early_stop=100)
    _, test_resnet50_adv_dataset = generate_adverasrial_examples(resnet50_premodel, test_loader, device=device2)
    _, train_resnet101_adv_dataset = generate_adverasrial_examples(resnet101_premodel, train_loader, device=device3, early_stop=100)
    _, test_resnet101_adv_dataset = generate_adverasrial_examples(resnet101_premodel, test_loader, device=device3)


    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer_vgg16 = optim.SGD(vgg16_model.parameters(), lr=0.001, momentum=0.9)
    optimizer_resnet50 = optim.SGD(resnet50_model.parameters(), lr=0.001, momentum=0.9)
    optimizer_resnet101 = optim.SGD(resnet101_model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    best_loss = np.inf
    early_stop_counter = 0
    patience = 3
    num_epochs = 100
    save_model_dir = "saved_model/models_ens_attack_pregen_training_final/"
    os.makedirs(save_model_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
        print(f"Training - epoch - {epoch+1}/{num_epochs}")
        logger.info(f"Training - epoch - {epoch+1}/{num_epochs}")
        vgg16_model.train()  # Set the model to training mode
        resnet50_model.train()  # Set the model to training mode
        resnet101_model.train()  # Set the model to training mode
        train_loss = 0
        correct = 0
        total = 0

        
        for train_adv_dataset in [train_vgg16_adv_dataset, train_resnet50_adv_dataset, train_resnet101_adv_dataset,]:
            batch = 0
            for images, labels in train_loader:
                print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
                print(f'batch: {batch+1}/{len(train_loader)}')
                # images = images.to(device)
                # labels = labels.to(device)
                adv_images, adv_labels = train_adv_dataset[batch]
                images = torch.cat([images.to(device4), adv_images.to(device4)], dim=0)
                labels = torch.cat([labels.to(device4), adv_labels.to(device4)], dim=0).to(device)

                optimizer_vgg16.zero_grad()
                outputs_vgg16 = vgg16_model(images.to(device1)).to(device)
                optimizer_resnet50.zero_grad()
                outputs_resnet50 = resnet50_model(images.to(device2)).to(device)
                optimizer_resnet101.zero_grad()
                outputs_resnet101 = resnet101_model(images.to(device3)).to(device)

                outputs = torch.stack((outputs_vgg16, outputs_resnet50, outputs_resnet101))
                loss = loss_fn(outputs.mean(dim=0), labels)
                loss.backward(retain_graph=True)
                optimizer_vgg16.step()
                optimizer_resnet50.step()
                optimizer_resnet101.step()

                train_loss += loss.item()

                _, predicted = torch.max(outputs.data.mean(dim=0), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                batch += 1
                if batch == 100:
                    break

            train_accuracy = correct / total



        # Evaluate on the test set
        vgg16_model.eval()  # Set the model to evaluation mode
        resnet50_model.eval()  # Set the model to evaluation mode
        resnet101_model.eval()  # Set the model to evaluation mode
        orig_correct = 0
        adv_correct = 0
        orig_total = 0
        adv_total = 0
        total = 0
        orig_val_loss = 0
        adv_val_loss = 0
        adv_val_loss_vgg16 = 0
        adv_val_loss_resnet50 = 0
        adv_val_loss_restnet101 = 0
        tot_val_loss = 0
        # batch = 0
        print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
        for images, labels in test_loader:
            # print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
            # print(f'batch: {batch+1}/{len(test_loader)}')

            # eval original
            # images = images.to(device)
            labels = labels.to(device)
            orig_total += labels.size(0)

            outputs_vgg16 = vgg16_model(images.to(device1)).to(device)
            outputs_resnet50 = resnet50_model(images.to(device2)).to(device)
            outputs_resnet101 = resnet101_model(images.to(device3)).to(device)
            outputs = torch.stack((outputs_vgg16, outputs_resnet50, outputs_resnet101))

            orig_val_loss = loss_fn(outputs.mean(dim=0), labels)
            _, predicted = torch.max(outputs.data.mean(dim=0), 1)
            tot_val_loss += orig_val_loss.item() * images.size(0)
            orig_val_loss += orig_val_loss.item() * images.size(0)
            orig_correct += (predicted == labels).sum().item()

            # eval adv
        for images, labels in test_vgg16_adv_dataset + test_resnet50_adv_dataset + test_resnet101_adv_dataset:
            # adv_images = images.to(device)
            # adv_labels = labels.to(device)

            adv_labels = labels.to(device)
            adv_total += adv_labels.size(0)

            outputs_vgg16 = vgg16_model(images.to(device1)).to(device)
            outputs_resnet50 = resnet50_model(images.to(device2)).to(device)
            outputs_resnet101 = resnet101_model(images.to(device3)).to(device)
            adv_outputs = torch.stack((outputs_vgg16, outputs_resnet50, outputs_resnet101))

            adv_val_loss_vgg16 = loss_fn(adv_outputs.mean(dim=0), adv_labels)
            _, adv_predicted = torch.max(adv_outputs.data.mean(dim=0), 1)
            tot_val_loss += adv_val_loss_vgg16.item() * images.size(0)
            adv_val_loss_vgg16 += adv_val_loss_vgg16.item() * images.size(0)
            adv_correct += (adv_predicted == adv_labels).sum().item()

        for images, labels in test_resnet50_adv_dataset:
            # adv_images = images.to(device)
            # adv_labels = labels.to(device)

            adv_labels = labels.to(device)
            adv_total += adv_labels.size(0)

            outputs_vgg16 = vgg16_model(images.to(device1)).to(device)
            outputs_resnet50 = resnet50_model(images.to(device2)).to(device)
            outputs_resnet101 = resnet101_model(images.to(device3)).to(device)
            adv_outputs = torch.stack((outputs_vgg16, outputs_resnet50, outputs_resnet101))

            adv_val_loss_resnet50 = loss_fn(adv_outputs.mean(dim=0), adv_labels)
            _, adv_predicted = torch.max(adv_outputs.data.mean(dim=0), 1)
            tot_val_loss += adv_val_loss_resnet50.item() * images.size(0)
            adv_val_loss_resnet50 += adv_val_loss_resnet50.item() * images.size(0)
            adv_correct += (adv_predicted == adv_labels).sum().item()

        for images, labels in test_resnet101_adv_dataset:
            # adv_images = images.to(device)
            # adv_labels = labels.to(device)

            adv_labels = labels.to(device)
            adv_total += adv_labels.size(0)

            outputs_vgg16 = vgg16_model(images.to(device1)).to(device)
            outputs_resnet50 = resnet50_model(images.to(device2)).to(device)
            outputs_resnet101 = resnet101_model(images.to(device3)).to(device)
            adv_outputs = torch.stack((outputs_vgg16, outputs_resnet50, outputs_resnet101))

            adv_val_loss_restnet101 = loss_fn(adv_outputs.mean(dim=0), adv_labels)
            _, adv_predicted = torch.max(adv_outputs.data.mean(dim=0), 1)
            tot_val_loss += adv_val_loss_restnet101.item() * images.size(0)
            adv_val_loss_restnet101 += adv_val_loss_restnet101.item() * images.size(0)
            adv_correct += (adv_predicted == adv_labels).sum().item()


        orig_val_accuracy = orig_correct / orig_total
        adv_val_accuracy = adv_correct / adv_total
        tot_val_accuracy = (orig_correct + adv_correct) / (orig_total + adv_total) 

        torch.save(vgg16_model, f'{save_model_dir}/vgg16_model_ep_{epoch+1}_attack_pregen_trained.pth')
        torch.save(resnet50_model, f'{save_model_dir}/resnet50_model_ep_{epoch+1}_attack_pregen_trained.pth')
        torch.save(resnet101_model, f'{save_model_dir}/resnet101_model_ep_{epoch+1}_attack_pregen_trained.pth')
        print(f"Epoch [{epoch+1}/{num_epochs}], train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, orig_val_loss: {orig_val_loss:.4f}, orig_val_accuracy: {orig_val_accuracy:.4f}, adv_val_loss: {adv_val_loss:.4f}, adv_val_accuracy: {adv_val_accuracy:.4f}, tot_val_loss: {tot_val_loss:.4f}, tot_val_accuracy: {tot_val_accuracy:.4f}")
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, orig_val_loss: {orig_val_loss:.4f}, orig_val_accuracy: {orig_val_accuracy:.4f}, adv_val_loss: {adv_val_loss:.4f}, adv_val_accuracy: {adv_val_accuracy:.4f}, tot_val_loss: {tot_val_loss:.4f}, tot_val_accuracy: {tot_val_accuracy:.4f}")

        if tot_val_loss < best_loss:
            best_loss = tot_val_loss
            early_stop_counter = 0
            # Save the best model weights
            best_model_weights_vgg16 = vgg16_model.state_dict()
            best_model_weights_resnet50 = resnet50_model.state_dict()
            best_model_weights_resnet101 = resnet101_model.state_dict()
        else:
            early_stop_counter += 1

        if epoch >= 80 and early_stop_counter >= patience:
            print(f"Early stopping at {epoch+1} after {patience} epochs without improvement.")
            break
    
    vgg16_model.load_state_dict(best_model_weights_vgg16)
    torch.save(vgg16_model, f'{save_model_dir}/vgg16_model_attack_pregen_best_model_final.pth')
    print(f"Saved {save_model_dir}/vgg16_model_attack_pregen_best_model_final.pth")
    resnet50_model.load_state_dict(best_model_weights_resnet50)
    torch.save(resnet50_model, f'{save_model_dir}/resnet50_model_attack_pregen_best_model_final.pth')
    print(f"Saved {save_model_dir}/resnet50_model_attack_pregen_best_model_final.pth")
    resnet101_model.load_state_dict(best_model_weights_resnet101)
    torch.save(resnet101_model, f'{save_model_dir}/resnet101_model_attack_pregen_best_model_final.pth')
    print(f"Saved {save_model_dir}/resnet101_model_attack_pregen_best_model_final.pth")


def model_evaluation_voting(models, test_dataset, test_adv_dataset, device=None):
    vgg16_model = models[0].to(device)
    resnet50_model = models[1].to(device)
    resnet101_model = models[2].to(device)
    # Evaluate on the test set
    vgg16_model.eval()  # Set the model to evaluation mode
    resnet50_model.eval()  # Set the model to evaluation mode
    resnet101_model.eval()  # Set the model to evaluation mode
    loss_fn = nn.CrossEntropyLoss()
    orig_correct = 0
    adv_correct = 0
    orig_total = 0
    adv_total = 0
    orig_val_loss = 0
    adv_val_loss = 0
    tot_val_loss = 0
    print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
    for images, labels in test_dataset:
        # print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
        # print(f'batch: {batch+1}/{len(test_loader)}')

        # eval original
        # images = images.to(device)
        labels = labels.to(device)
        orig_total += labels.size(0)

        outputs_vgg16 = vgg16_model(images.to(device))
        outputs_resnet50 = resnet50_model(images.to(device))
        outputs_resnet101 = resnet101_model(images.to(device))
        outputs = torch.stack((outputs_vgg16, outputs_resnet50, outputs_resnet101))

        orig_val_loss = loss_fn(outputs.mean(dim=0), labels)
        _, predicted = torch.max(outputs.data.mean(dim=0), 1)
        tot_val_loss += orig_val_loss.item() * images.size(0)
        orig_val_loss += orig_val_loss.item() * images.size(0)
        orig_correct += (predicted == labels).sum().item()

        # eval adv
    for images, labels in test_adv_dataset:
        # adv_images = images.to(device)
        # adv_labels = labels.to(device)

        adv_labels = labels.to(device)
        adv_total += adv_labels.size(0)

        outputs_vgg16 = vgg16_model(images.to(device))
        outputs_resnet50 = resnet50_model(images.to(device))
        outputs_resnet101 = resnet101_model(images.to(device))
        adv_outputs = torch.stack((outputs_vgg16, outputs_resnet50, outputs_resnet101))

        adv_val_loss = loss_fn(adv_outputs.mean(dim=0), adv_labels)
        _, adv_predicted = torch.max(adv_outputs.data.mean(dim=0), 1)
        tot_val_loss += adv_val_loss.item() * images.size(0)
        adv_val_loss += adv_val_loss.item() * images.size(0)
        adv_correct += (adv_predicted == adv_labels).sum().item()


    orig_val_accuracy = orig_correct / orig_total
    adv_val_accuracy = adv_correct / adv_total
    tot_val_accuracy = (orig_correct + adv_correct) / (orig_total + adv_total)

    print(f"orig_val_loss: {orig_val_loss:.4f}, orig_val_accuracy: {orig_val_accuracy:.4f}, adv_val_loss: {adv_val_loss:.4f}, adv_val_accuracy: {adv_val_accuracy:.4f}, tot_val_loss: {tot_val_loss:.4f}, tot_val_accuracy: {tot_val_accuracy:.4f}")
    logger.info(f"orig_val_loss: {orig_val_loss:.4f}, orig_val_accuracy: {orig_val_accuracy:.4f}, adv_val_loss: {adv_val_loss:.4f}, adv_val_accuracy: {adv_val_accuracy:.4f}, tot_val_loss: {tot_val_loss:.4f}, tot_val_accuracy: {tot_val_accuracy:.4f}")



def model_evaluation():
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

    ens_models_vgg16_adv_trained = torch.load('saved_model/models_ens_attack_pregen_training_final//vgg16_model_attack_pregen_best_model_final.pth')
    ens_models_resnet50_adv_trained = torch.load('saved_model/models_ens_attack_pregen_training_final//resnet50_model_attack_pregen_best_model_final.pth')
    ens_models_resnet101_adv_trained = torch.load('saved_model/models_ens_attack_pregen_training_final//resnet101_model_attack_pregen_best_model_final.pth')

    device = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:2")
    device3 = torch.device("cuda:3")

    vgg16_pretrained_model = torch.load('saved_model/vgg16_best_model.pth')
    resnet50_pretrained_model = torch.load('saved_model/resnet50_best_model.pth')
    resnet101_pretrained_model = torch.load('saved_model/resnet101_best_model.pth')

    orig_test_dataset, vgg16_adv_dataset = generate_adverasrial_examples(vgg16_pretrained_model, test_loader, device=device1)
    orig_test_dataset, resnet50_adv_dataset = generate_adverasrial_examples(resnet50_pretrained_model, test_loader, device=device2)
    orig_test_dataset, resnet101_adv_dataset = generate_adverasrial_examples(resnet101_pretrained_model, test_loader, device=device3)


    models_list = [ens_models_vgg16_adv_trained, ens_models_resnet50_adv_trained, ens_models_resnet101_adv_trained]

    print("Evaluation on vgg16_adv_dataset")
    model_evaluation_voting(models_list, test_loader, vgg16_adv_dataset, device=device3)

    print("Evaluation on resnet50_adv_dataset")
    model_evaluation_voting(models_list, test_loader, resnet50_adv_dataset, device=device3)

    print("Evaluation on resnet101_adv_dataset")
    model_evaluation_voting(models_list, test_loader, resnet101_adv_dataset, device=device3)

model_evaluation()




# models_ensemble_attack_pregen_training()