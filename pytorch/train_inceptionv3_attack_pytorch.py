import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.models import inception_v3

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
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
        logging.FileHandler(f'log_files/logfile_inceptionv3_{cur_time}.log'),  # Specify the path to the log file
    ]
)

# Create a logger instance
logger = logging.getLogger()

"""The Projected Gradient Descent attack."""
import numpy as np
import torch

from cleverhans.torch.utils import clip_eta
from cleverhans.torch.utils import optimize_linear


def modified_fast_gradient_method(
    model_fn,
    x,
    eps,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
):
    """
    PyTorch implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    # Compute loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(model_fn(x).logits, y)
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.backward()
    optimal_perturbation = optimize_linear(x.grad, eps, norm)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x
 

def modified_projected_gradient_descent(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=True,
    rand_minmax=None,
    sanity_checks=True,
):
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to False. or the
    Madry et al. (2017) method if rand_init is set to True.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
    :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
              which the random perturbation on x was drawn. Effective only when rand_init is
              True. Default equals to eps.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if eps_iter < 0:
        raise ValueError(
            "eps_iter must be greater than or equal to 0, got {} instead".format(
                eps_iter
            )
        )
    if eps_iter == 0:
        return x

    assert eps_iter <= eps, (eps_iter, eps)
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # Initialize loop variables
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
    else:
        eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x).logits, 1)

    i = 0
    while i < nb_iter:
        adv_x = modified_fast_gradient_method(
            model_fn,
            adv_x,
            eps_iter,
            norm,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
        )

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast clip_min and clip_max to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


def inceptionv3_attack_training():
    # Define the Resmodel model
    model = torch.load('saved_model/inceptionv3_pdg_training/inceptionv3_ep_3_pdg_trained.pth')

    # Load and preprocess the MNIST dataset
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

    # Set the device to use (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")
    print("device: ", device)
    logger.info("device:")
    logger.info(device.type)
    model = model.to(device)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    best_loss = np.inf
    early_stop_counter = 0
    best_model_weights = None
    patience = 3
    num_epochs = 50
    save_model_dir = "saved_model/inceptionv3_pdg_training/"
    os.makedirs(save_model_dir, exist_ok=True)
    for epoch in range(3, num_epochs):
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

            images = modified_projected_gradient_descent(model, images, eps=0.3, eps_iter=0.01, nb_iter=40, norm=np.inf)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch += 1

        train_accuracy = correct / total

        # Evaluate on the test set
        print(f"Eval - epoch - {epoch+1}/{num_epochs}")
        logger.info(f"Eval - epoch - {epoch+1}/{num_epochs}")
        model.eval()  # Set the model to evaluation mode
        correct = 0
        adv_correct = 0
        total = 0
        val_loss = 0
        adv_val_loss = 0
        batch = 0
        for images, labels in test_loader:
            print('Time: ', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
            print(f'batch: {batch+1}/{len(test_loader)}')
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_loss = loss_fn(outputs, labels)
            val_loss += val_loss.item() * images.size(0)
            correct += (predicted == labels).sum().item()

            adv_images = projected_gradient_descent(model, images, eps=0.3, eps_iter=0.01, nb_iter=40, norm=np.inf)
            adv_outputs = model(adv_images)
            _, adv_predicted = torch.max(adv_outputs.data, 1)
            adv_val_loss = loss_fn(adv_outputs, labels)
            adv_val_loss += adv_val_loss.item() * images.size(0)
            adv_correct += (adv_predicted == labels).sum().item()

            batch += 1

        val_accuracy = correct / total
        adv_val_accuracy = adv_correct / total

        torch.save(model, f'{save_model_dir}/inceptionv3_ep_{epoch+1}_pdg_trained.pth')
        print(f"Saved {save_model_dir}/inceptionv3_ep_{epoch+1}_pdg_trained.pth")
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
    torch.save(model, f'{save_model_dir}/inceptionv3_best_model_pdg_trained.pth')
    print(f"Saved {save_model_dir}/inceptionv3_best_model_pdg_trained.pth")