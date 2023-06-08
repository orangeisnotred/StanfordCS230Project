import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

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


def generate_adverasrial_examples(model, dataset, device=None, resize=None):
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

    return orig_dataset, adv_dataset


def model_evaluation(model, test_dataset, test_adv_dataset, device=None):
    
    # Evaluate on the test set
    model.eval()  
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


    orig_val_accuracy = orig_correct / orig_total
    adv_val_accuracy = adv_correct / adv_total
    tot_val_accuracy = (orig_correct + adv_correct) / (orig_total + adv_total)

    print(f"orig_val_loss: {orig_val_loss:.4f}, orig_val_accuracy: {orig_val_accuracy:.4f}, adv_val_loss: {adv_val_loss:.4f}, adv_val_accuracy: {adv_val_accuracy:.4f}, tot_val_loss: {tot_val_loss:.4f}, tot_val_accuracy: {tot_val_accuracy:.4f}")
    logger.info(f"orig_val_loss: {orig_val_loss:.4f}, orig_val_accuracy: {orig_val_accuracy:.4f}, adv_val_loss: {adv_val_loss:.4f}, adv_val_accuracy: {adv_val_accuracy:.4f}, tot_val_loss: {tot_val_loss:.4f}, tot_val_accuracy: {tot_val_accuracy:.4f}")


def model_evaluation_voting(models, test_dataset, test_adv_dataset, device=None):
    vgg16_model = models[0]
    resnet50_model = models[1]
    resnet101_model = models[2]
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
    for images, labels in test_adv_dataset:
        # adv_images = images.to(device)
        # adv_labels = labels.to(device)

        adv_labels = labels.to(device)
        adv_total += adv_labels.size(0)

        outputs_vgg16 = vgg16_model(images.to(device1)).to(device)
        outputs_resnet50 = resnet50_model(images.to(device2)).to(device)
        outputs_resnet101 = resnet101_model(images.to(device3)).to(device)
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
device = torch.device("cuda:0")
device1 = torch.device("cuda:1")
device2 = torch.device("cuda:2")
device3 = torch.device("cuda:3")
device4 = torch.device("cuda:4")
device5 = torch.device("cuda:5")
device6 = torch.device("cuda:6")
device7 = torch.device("cuda:7")


vgg16_pretrained_model = torch.load('final_models/vgg16_pretrained.pth')
resnet50_pretrained_model = torch.load('final_models/resnet50_pretrained.pth')
resnet101_pretrained_model = torch.load('final_models/resnet101_pretrained.pth')
# 

# vgg16_pretrained_model = torch.load('final_models/vgg16_gen_adv.pth')
# resnet50_pretrained_model = torch.load('final_models/resnet50_gen_adv.pth')
# resnet101_pretrained_model = torch.load('final_models/resnet101_gen_adv.pth')

vgg16_adv_trained_model = torch.load('final_models/vgg16_adv_trained_final.pth')
resnet50_adv_trained_model = torch.load('final_models/resnet50_adv_trained_final.pth')
resnet101_adv_trained_model = torch.load('final_models/resnet101_adv_trained_final.pth')
ens_vgg16_adv_trained_model = torch.load('final_models/ens_vgg16_adv_trained.pth')
ens_resnet50_adv_trained_model = torch.load('final_models/ens_resnet50_adv_trained_final.pth')
ens_models_vgg16_adv_trained = torch.load('final_models/ens_models_vgg16_adv_trained_final.pth')
ens_models_resnet50_adv_trained = torch.load('final_models/ens_models_resnet50_adv_trained_final.pth')
ens_models_resnet101_adv_trained = torch.load('final_models/ens_models_resnet101_adv_trained_final.pth')

vgg16_pretrained_model = vgg16_pretrained_model.to(device1)
resnet50_pretrained_model = resnet50_pretrained_model.to(device2)
resnet101_pretrained_model = resnet101_pretrained_model.to(device3)

vgg16_adv_trained_model = vgg16_adv_trained_model.to(device1)
resnet50_adv_trained_model = resnet50_adv_trained_model.to(device2)
resnet101_adv_trained_model = resnet101_adv_trained_model.to(device3)
ens_vgg16_adv_trained_model = ens_vgg16_adv_trained_model.to(device)
ens_resnet50_adv_trained_model = ens_resnet50_adv_trained_model.to(device)
ens_models_vgg16_adv_trained = ens_models_vgg16_adv_trained.to(device1)
ens_models_resnet50_adv_trained = ens_models_resnet50_adv_trained.to(device2)
ens_models_resnet101_adv_trained = ens_models_resnet101_adv_trained.to(device3)




orig_test_dataset, vgg16_adv_dataset = generate_adverasrial_examples(vgg16_pretrained_model, test_loader, device=device1)
orig_test_dataset, resnet50_adv_dataset = generate_adverasrial_examples(resnet50_pretrained_model, test_loader, device=device2)
orig_test_dataset, resnet101_adv_dataset = generate_adverasrial_examples(resnet101_pretrained_model, test_loader, device=device3)

# vgg16_adv_dataset = generate_adverasrial_examples(vgg16_adv_trained_model, orig_test_dataset, device=device1)
# resnet50_adv_dataset = generate_adverasrial_examples(resnet50_adv_trained_model, orig_test_dataset, device=device2)
# resnet101_adv_dataset = generate_adverasrial_examples(resnet101_adv_trained_model, orig_test_dataset, device=device3)


transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)


inceptionv3_pretrained_model = torch.load('/mnt/task_runtime/saved_model/inceptionv3_best_model.pth')
inceptionv3_pretrained_model = inceptionv3_pretrained_model.to(device4)
orig_test_dataset, inceptionv3_adv_dataset = generate_adverasrial_examples(inceptionv3_pretrained_model, test_loader, device=device4, resize=(224,224))

models_list = [ens_models_vgg16_adv_trained, ens_models_resnet50_adv_trained, ens_models_resnet101_adv_trained]

print("Evaluation on vgg16_adv_dataset")
print("Evaluate vgg16_adv_trained_model")
model_evaluation(vgg16_adv_trained_model.to(device1), orig_test_dataset, vgg16_adv_dataset, device=device1)
print("Evaluate resnet50_adv_trained_model")
model_evaluation(resnet50_adv_trained_model.to(device1), orig_test_dataset, vgg16_adv_dataset, device=device1)
print("Evaluate resnet101_adv_trained_model")
model_evaluation(resnet101_adv_trained_model.to(device1), orig_test_dataset, vgg16_adv_dataset, device=device1)
print("Evaluate ens_vgg16_adv_trained_model")
model_evaluation(ens_vgg16_adv_trained_model.to(device1), orig_test_dataset, vgg16_adv_dataset, device=device1)
print("Evaluate ens_resnet_adv_trained_model")
model_evaluation(ens_resnet50_adv_trained_model.to(device1), orig_test_dataset, vgg16_adv_dataset, device=device1)
print("Evaluate ens_models_adv_trained_model")
model_evaluation_voting(models_list, orig_test_dataset, vgg16_adv_dataset, device=device1)
print("")
print("")
print("")

print("Evaluation on resnet50_adv_dataset")
print("Evaluate vgg16_adv_trained_model")
model_evaluation(vgg16_adv_trained_model.to(device2), orig_test_dataset, resnet50_adv_dataset, device=device2)
print("Evaluate resnet50_adv_trained_model")
model_evaluation(resnet50_adv_trained_model.to(device2), orig_test_dataset, resnet50_adv_dataset, device=device2)
print("Evaluate resnet101_adv_trained_model")
model_evaluation(resnet101_adv_trained_model.to(device2), orig_test_dataset, resnet50_adv_dataset, device=device2)
print("Evaluate ens_vgg16_adv_trained_model")
model_evaluation(ens_vgg16_adv_trained_model.to(device2), orig_test_dataset, resnet50_adv_dataset, device=device2)
print("Evaluate ens_resnet_adv_trained_model")
model_evaluation(ens_resnet50_adv_trained_model.to(device2), orig_test_dataset, resnet50_adv_dataset, device=device2)
print("Evaluate ens_models_adv_trained_model")
model_evaluation_voting(models_list, orig_test_dataset, resnet50_adv_dataset, device=device2)
print("")
print("")
print("")

print("Evaluation on resnet101_adv_dataset")
print("Evaluate vgg16_adv_trained_model")
model_evaluation(vgg16_adv_trained_model.to(device3), orig_test_dataset, resnet101_adv_dataset, device=device3)
print("Evaluate resnet50_adv_trained_model")
model_evaluation(resnet50_adv_trained_model.to(device3), orig_test_dataset, resnet101_adv_dataset, device=device3)
print("Evaluate resnet101_adv_trained_model")
model_evaluation(resnet101_adv_trained_model.to(device3), orig_test_dataset, resnet101_adv_dataset, device=device3)
print("Evaluate ens_vgg16_adv_trained_model")
model_evaluation(ens_vgg16_adv_trained_model.to(device3), orig_test_dataset, resnet101_adv_dataset, device=device3)
print("Evaluate ens_resnet_adv_trained_model")
model_evaluation(ens_resnet50_adv_trained_model.to(device3), orig_test_dataset, resnet101_adv_dataset, device=device3)
print("Evaluate ens_models_adv_trained_model")
model_evaluation_voting(models_list, orig_test_dataset, resnet101_adv_dataset, device=device3)
print("")
print("")
print("")


print("Evaluation on inceptionv3_adv_dataset")
print("Evaluate vgg16_adv_trained_model")
model_evaluation(vgg16_adv_trained_model.to(device4), orig_test_dataset, inceptionv3_adv_dataset, device=device4)
print("Evaluate resnet50_adv_trained_model")
model_evaluation(resnet50_adv_trained_model.to(device4), orig_test_dataset, inceptionv3_adv_dataset, device=device4)
print("Evaluate resnet101_adv_trained_model")
model_evaluation(resnet101_adv_trained_model.to(device4), orig_test_dataset, inceptionv3_adv_dataset, device=device4)
print("Evaluate ens_vgg16_adv_trained_model")
model_evaluation(ens_vgg16_adv_trained_model.to(device4), orig_test_dataset, inceptionv3_adv_dataset, device=device4)
print("Evaluate ens_resnet_adv_trained_model")
model_evaluation(ens_resnet50_adv_trained_model.to(device4), orig_test_dataset, inceptionv3_adv_dataset, device=device4)
print("Evaluate ens_models_adv_trained_model")
model_evaluation_voting(models_list, orig_test_dataset, inceptionv3_adv_dataset, device=device4)
print("")
print("")
print("")



print('Done')