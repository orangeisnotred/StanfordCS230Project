import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications import VGG16, ResNet50, ResNet101
from efficientnet.tfkeras import EfficientNetB5
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import pdb

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

import logging
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'log_files/logfile_resnet50_{datetime.datetime.now()}.log'),  # Specify the path to the log file
    ]
)
os.makedirs('log_files', exist_ok = True)
# Create a logger instance
logger = logging.getLogger()

# os.environ["TF_METAL_ENABLED"] = "1"
gpus = tf.config.list_physical_devices('GPU')
print("GPU available: ", gpus)

class MyDataset:
    def __init__(self, dataset, target_img_size, train_exmaple_counts=None, test_example_counts=None, processed_data_dir=None):
        self.dataset = dataset
        self.target_size = target_img_size
        self.train_exmaple_counts = train_exmaple_counts
        self.test_example_counts = test_example_counts
        self.processed_data_dir = processed_data_dir

        self.train_images = None    
        self.train_labels = None
        self.validation_images = None
        self.validation_labels = None
        self.test_images = None
        self.test_labels = None
        self.num_classes = None

        self.train_images_fgsm_vgg16 = None
        self.train_images_pgd_vgg16 = None
        self.train_images_fgsm_effnetB5 = None
        self.train_images_pgd_effnetB5 = None
        self.train_images_fgsm_resnet50 = None
        self.train_images_pgd_resnet50 = None

        self.test_images_fgsm_vgg16 = None
        self.test_images_pgd_vgg16 = None
        self.test_images_fgsm_effnetB5 = None
        self.test_images_pgd_effnetB5 = None
        self.test_images_fgsm_resnet50 = None
        self.test_images_pgd_resnet50 = None
        pdb.set_trace()
        
        if self.dataset == 'preprocessed':
            self.load_processed_data(self.processed_data_dir)
        elif self.dataset == 'mnist':
            self.load_mnist()
            self.pre_process_data(train_exmaple_counts=self.train_exmaple_counts, test_example_counts=self.test_example_counts)


    def load_mnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

        # Process the data, normalize it, reshape it, and one-hot-encode the labels
        img_rows, img_cols, channels = 28, 28, 1
        self.num_classes = 10

        self.train_images= x_train / 255
        self.test_images = x_test / 255

        self.train_labels = tf.keras.utils.to_categorical(y_train, self.num_classes)
        self.test_labels = tf.keras.utils.to_categorical(y_test, self.num_classes)

        print("Data shapes", self.train_images.shape, self.train_labels.shape, self.test_images.shape, self.test_labels.shape)

    def load_processed_data(self, processed_data_dir):
        self.train_images = np.load(os.path.join(processed_data_dir, 'train_images.npy'))
        print("Load from: ", os.path.join(processed_data_dir, 'train_images.npy'))
        self.val_images = np.load(os.path.join(processed_data_dir, 'val_images.npy'))
        print("Load from: ", os.path.join(processed_data_dir, 'val_images.npy'))
        self.test_images = np.load(os.path.join(processed_data_dir, 'test_images.npy'))
        print("Load from: ", os.path.join(processed_data_dir, 'test_images.npy'))
        self.train_labels = np.load(os.path.join(processed_data_dir, 'train_labels.npy'))
        print("Load from: ", os.path.join(processed_data_dir, 'train_labels.npy'))
        self.val_labels = np.load(os.path.join(processed_data_dir, 'val_labels.npy'))
        print("Load from: ", os.path.join(processed_data_dir, 'val_labels.npy'))
        self.test_labels = np.load(os.path.join(processed_data_dir, 'test_labels.npy'))
        print("Load from: ", os.path.join(processed_data_dir, 'test_labels.npy'))

        path = os.path.join(processed_data_dir, 'train_images_fgsm_vgg16.npy')
        if os.path.exists(path):
            self.train_images_fgsm_vgg16 = np.load(path)
            print("Load from: ", path)
        path = os.path.join(processed_data_dir, 'train_images_pgd_vgg16.npy')
        if os.path.exists(path):
            self.train_images_pgd_vgg16 = np.load(path)
            print("Load from: ", path)
        path = os.path.join(processed_data_dir, 'train_images_fgsm_effnetB5.npy')
        if os.path.exists(path):
            self.train_images_fgsm_effnetB5 = np.load(path)
            print("Load from: ", path)
        path = os.path.join(processed_data_dir, 'train_images_pgd_effnetB5.npy')
        if os.path.exists(path):
            self.train_images_pgd_effnetB5 = np.load(path)
            print("Load from: ", path)
        path = os.path.join(processed_data_dir, 'train_images_fgsm_resnet50.npy')
        if os.path.exists(path):
            self.train_images_fgsm_resnet50 = np.load(path)
            print("Load from: ", path)
        path = os.path.join(processed_data_dir, 'train_images_pgd_resnet50.npy')
        if os.path.exists(path):
            self.train_images_pgd_resnet50 = np.load(path)
            print("Load from: ", path)
        
        
        path = os.path.join(processed_data_dir, 'test_images_fgsm_vgg16.npy')
        if os.path.exists(path):
            self.test_images_fgsm_vgg16 = np.load(path)
            print("Load from: ", path)
        path = os.path.join(processed_data_dir, 'test_images_pgd_vgg16.npy')
        if os.path.exists(path):
            self.test_images_pgd_vgg16 = np.load(path)
            print("Load from: ", path)
        path = os.path.join(processed_data_dir, 'test_images_fgsm_effnetB5.npy')
        if os.path.exists(path):
            self.test_images_fgsm_effnetB5 = np.load(path)
            print("Load from: ", path)
        path = os.path.join(processed_data_dir, 'test_images_pgd_effnetB5.npy')
        if os.path.exists(path):
            self.test_images_pgd_effnetB5 = np.load(path)
            print("Load from: ", path)
        path = os.path.join(processed_data_dir, 'test_images_fgsm_resnet50.npy')
        if os.path.exists(path):
            self.test_images_fgsm_resnet50 = np.load(path)
            print("Load from: ", path)
        path = os.path.join(processed_data_dir, 'test_images_pgd_resnet50.npy')
        if os.path.exists(path):
            self.test_images_pgd_resnet50 = np.load(path)
            print("Load from: ", path)
        
    
    def pre_process_data(self, train_exmaple_counts=None, test_example_counts=None, val_percentage=0.2):
        if not train_exmaple_counts:
            train_exmaple_counts = self.train_images.shape[0]
        self.train_images = self.train_images[:train_exmaple_counts, ...]
        self.train_labels = self.train_labels[:train_exmaple_counts, ...]

        if os.path.exists("val_indices.csv"):
            val_indices = list(pd.read_csv("val_indices.csv")['ids'])
            train_indices = list(set(range(len(self.train_images))) - set(val_indices))
        else:
            validation_size = int(val_percentage * len(self.train_images))
            val_indices = tf.random.shuffle(tf.range(0, len(self.train_images)))[:validation_size].numpy().tolist()
            train_indices = list(set(range(len(self.train_images))) - set(val_indices))
        
        gpus = tf.config.list_logical_devices('GPU')
        print('gpus: ', gpus)
        with tf.device(gpus[0].name):
            self.val_data = []
            for index in val_indices:
                image = cv2.resize(self.train_images[index], self.target_size, interpolation=cv2.INTER_LINEAR)
                rgb_image = np.stack((image,) * 3, axis=-1)
                self.val_data.append(rgb_image)
            self.val_images = np.array(self.val_data)
            self.val_labels = self.train_labels[val_indices]
            logging.info(f"Val data ready")

            self.train_data = []
            for index in train_indices:
                image = cv2.resize(self.train_images[index], self.target_size, interpolation=cv2.INTER_LINEAR)
                rgb_image = np.stack((image,) * 3, axis=-1)
                self.train_data.append(rgb_image)
            self.train_images = np.array(self.train_data)
            self.train_labels = self.train_labels[train_indices]
            logging.info(f"Train data ready")

            if not test_example_counts:
                test_example_counts = self.test_images.shape[0]
            self.test_images = self.test_images[:test_example_counts, ...]
            self.test_labels = self.test_labels[:test_example_counts, ...]

            self.test_data = []
            for img in self.test_images:
                image = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
                rgb_image = np.stack((image,) * 3, axis=-1)
                self.test_data.append(rgb_image)
            self.test_images = np.array(self.test_data)
            logging.info(f"Test data ready")

            
            dir_path = f"dataset_{self.dataset}_train_{train_exmaple_counts}_test_{test_example_counts}"
            logging.info(f"make dir: {dir_path}")
            os.makedirs(dir_path, exist_ok = True)

            save_path = os.path.join(dir_path, "train_images.npy")
            print("Save to", save_path)
            np.save(save_path, self.train_images)

            save_path = os.path.join(dir_path, "val_images.npy")
            print("Save to", save_path)
            np.save(save_path, self.val_images)

            save_path = os.path.join(dir_path, "test_images.npy")
            print("Save to", save_path)
            np.save(save_path, self.test_images)

            save_path = os.path.join(dir_path, "train_labels.npy")
            print("Save to", save_path)
            np.save(save_path, self.train_labels)

            save_path = os.path.join(dir_path, "val_labels.npy")
            print("Save to", save_path)
            np.save(save_path, self.val_labels)

            save_path = os.path.join(dir_path, "test_labels.npy")
            print("Save to", save_path)
            np.save(save_path, self.test_labels)

    def random_split_train_validation_set(self, data, labels, batch_size, random_state=0):
        
        validation_size = int(0.2 * len(data))
        val_indices = tf.random.shuffle(tf.range(0, len(data)))[:validation_size].numpy().tolist()
        train_indices = list(set(range(len(data))) - set(val_indices))
        val_data = data[val_indices] 
        val_labels = labels[val_indices] 
        train_data = np.zeros((48000, 224, 224, 3))
        train_data[:12000] = data[train_indices[:12000]]
        train_data[12000:24000] = data[train_indices[12000:24000]]
        train_data[24000:36000] = data[train_indices[24000:36000]]
        train_data[36000:48000] = data[train_indices[36000:48000]]
        train_labels = labels[train_indices]

        print("Save to", 'dataset_mnist_train_60000_test_10000/validation_images.npy')
        np.save('dataset_mnist_train_60000_test_10000/validation_images.npy', val_data)

        print("Save to", 'dataset_mnist_train_60000_test_10000/validation_labels.npy')
        np.save('dataset_mnist_train_60000_test_10000/validation_labels.npy', val_labels)
        

        print("Save to", 'dataset_mnist_train_60000_test_10000/train_images.npy')
        np.save('dataset_mnist_train_60000_test_10000/train_images.npy', train_data)

        print("Save to", 'dataset_mnist_train_60000_test_10000/train_labels.npy')
        np.save('dataset_mnist_train_60000_test_10000/train_labels.npy', train_labels)



        

        
        batch_size = 1000
        num = int(len(data.train_labels) / batch_size)
        for n in range(num):
            print(f"Batch - {n + 1}/{num}")
            
            resnet50.model.fit(mydata.train_images[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=50, batch_size=128)


        train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=random_state)



        
        
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = dataset.shuffle(len(data), reshuffle_each_iteration=False)

        train_dataset = dataset.skip(validation_size)
        val_dataset = dataset.take(validation_size)

        # num_samples = train_data.shape[0]

        # Create a random permutation of indices
        indices = tf.range(train_data.shape[0])
        shuffled_indices = tf.random.shuffle(indices)

        # # Shuffle the data and labels using the shuffled indices
        # shuffled_data = tf.gather(data, shuffled_indices)
        # shuffled_labels = tf.gather(labels, shuffled_indices)


        train_iterator = train_dataset.batch(batch_size).prefetch(1).make_one_shot_iterator()
        val_iterator = val_dataset.batch(batch_size).prefetch(1).make_one_shot_iterator()

        return train_data, val_data, train_labels, val_labels, shuffled_indices


class MyModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.name = None
        self.model = None
    
    def set_model_name(self, name):
        self.name = name
    
    def early_stopping(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss for early stopping
        patience=5,           # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Restore the weights of the best-performing model
    )

    def preload_vgg16(self, weights='imagenet', include_top=False):
        vgg_model = VGG16(weights=weights, include_top=include_top, input_shape=(224, 224, 3))

        # Freeze the pre-trained layers to prevent further training
        for layer in vgg_model.layers:
            layer.trainable = False

        # Add your own classification layers on top of the pre-trained model
        self.model = tf.keras.Sequential([
            vgg_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')  # Replace num_classes with the number of your classes
        ])

        # Compile and train your model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def preload_effnetb5(self, weights='imagenet',include_top=False):
        effnet_model = EfficientNetB5(weights=weights,include_top=include_top, input_shape=(224, 224, 3))
       
        # Freeze the base model layers
        effnet_model.trainable = False

        # Add your own classification layers on top of the pre-trained model
        self.model = tf.keras.Sequential([
            effnet_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')  # Replace num_classes with the number of your classes
        ])

        # Compile and train your model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def preload_resnet50(self, weights='imagenet',include_top=False):
        resnet_model = ResNet50(weights=weights,include_top=include_top, input_shape=(224, 224, 3))
       
        # Freeze the base model layers
        resnet_model.trainable = False

        # Add your own classification layers on top of the pre-trained model
        self.model = tf.keras.Sequential([
            resnet_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')  # Replace num_classes with the number of your classes
        ])

        # Compile and train your model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def preload_resnet101(self, weights='imagenet',include_top=False):
        resnet_model = ResNet101(weights=weights,include_top=include_top, input_shape=(224, 224, 3))
       
        # Freeze the base model layers
        resnet_model.trainable = False

        # Add your own classification layers on top of the pre-trained model
        self.model = tf.keras.Sequential([
            resnet_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')  # Replace num_classes with the number of your classes
        ])

        # Compile and train your model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def generate_adversarial_images_with_fgsm(model, data, output_file_name, eps=0.1, batch_size=100, dir_path="dataset_mnist_train_60000_test_10000"):
    num = int(len(data) / batch_size)
    adversarial_images = []
    for n in range(num):
        print(f"Generating adversarial images with fgsm - {n + 1}/{num}")
        generated_imgs = fast_gradient_method(model, data[batch_size * n: batch_size * (n + 1)], eps, np.inf)
        if not len(adversarial_images):
            adversarial_images = generated_imgs
        else:
            adversarial_images = np.vstack((adversarial_images, generated_imgs))
    
    os.makedirs(dir_path, exist_ok = True)
    save_path = os.path.join(dir_path, output_file_name)
    print("Save to", save_path)
    np.save(save_path, adversarial_images)
    return adversarial_images

def generate_adversarial_images_with_pgd(model, data, output_file_name, eps=0.1, eps_iter=0.01, nb_iter=2, batch_size=100, dir_path="dataset_mnist_train_60000_test_10000"):
    num = int(len(data) / batch_size)
    adversarial_images = []
    for n in range(num):
        print(f"Generating adversarial images with pgd - {n + 1}/{num}")
        generated_imgs = projected_gradient_descent(model, data[batch_size * n: batch_size * (n + 1)], eps, eps_iter, nb_iter, np.inf)
        if not len(adversarial_images):
            adversarial_images = generated_imgs
        else:
            adversarial_images = np.vstack((adversarial_images, generated_imgs))
    
    os.makedirs(dir_path, exist_ok = True)
    save_path = os.path.join(dir_path, output_file_name)
    print("Save to", save_path)
    np.save(save_path, adversarial_images)
    return adversarial_images


def vgg16_training(mydata, preload_model=None, model_name_set_to=None, model_save_to='saved_models_no_given_dir', generate_images_dir_path=None, train_model_clean=False, run_eval_clean=False, run_eval_fgsm=False, run_eval_pgd=False, 
                   generate_images_fgsm=False, generate_images_pgd=False, train_model_fgsm=False, train_model_pgd=False):
    vgg16 = MyModel(10)
    vgg16.preload_vgg16()
    vgg16.set_model_name(model_name_set_to)

    if preload_model:
        # load trained vgg16 with mnist
        vgg16.model=load_model(preload_model)
        vgg16.model.summary()
        print(f'{preload_model} Model Loaded!')

    if train_model_clean:
        # train preload vgg16 with mnist
        batch_size = 1000
        num = int(len(mydata.train_labels) / batch_size)
        for n in range(num):
            print(f"Training - {n + 1}/{num}")
            vgg16.model.fit(mydata.train_images[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=20, batch_size=128)
        
        # save model
        vgg16.model.save(f'{model_save_to}/mnist_vgg16_model.h5')
        print('mnist_vgg16_model Model Saved!')

        # save model
        vgg16.model.save_weights(f'{model_save_to}/mnist_vgg16_weights.h5')
        print('mnist_vgg16_model Weights Saved!')

    if generate_images_fgsm:
        mydata.train_images_fgsm_vgg16 = generate_adversarial_images_with_fgsm(vgg16.model, mydata.train_images, "train_images_fgsm_vgg16.npy", dir_path=generate_images_dir_path)
        mydata.test_images_fgsm_vgg16 = generate_adversarial_images_with_fgsm(vgg16.model, mydata.test_images, "test_images_fgsm_vgg16.npy", dir_path=generate_images_dir_path)
         # mydata.train_images_fgsm_vgg16 = generate_adversarial_images_with_fgsm(vgg16.model, mydata.train_images[30201:, ...], "train_images_fgsm_vgg16.npy")

    if train_model_fgsm:
        # train preload vgg16 with mnist
        batch_size = 1000
        num = int(len(mydata.train_labels) / batch_size)
        for n in range(num):
            print(f"Training - {n + 1}/{num}")
            vgg16.model.fit(mydata.train_images_fgsm_vgg16[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=1, batch_size=128)
        
        # save model
        vgg16.model.save(f'{model_save_to}/mnist_vgg16_model_with_fgsm_vgg16.h5')
        print('mnist_vgg16_model_with_fgsm_vgg16 Model Saved!')

        # save model
        vgg16.model.save_weights(f'{model_save_to}/mnist_vgg16_weights_with_fgsm_vgg16.h5')
        print('mnist_vgg16_model_with_fgsm_vgg16 Weights Saved!')

    if generate_images_pgd:
        mydata.train_images_pgd_vgg16 = generate_adversarial_images_with_pgd(vgg16.model, mydata.train_images, "train_images_pgd_vgg16.npy", dir_path=generate_images_dir_path)
        mydata.test_images_pgd_vgg16 = generate_adversarial_images_with_pgd(vgg16.model, mydata.test_images, "test_images_pgd_vgg16.npy", dir_path=generate_images_dir_path)

    if train_model_pgd:
        # train preload vgg16 with mnist
        batch_size = 1000
        num = int(len(mydata.train_labels) / batch_size)
        for n in range(num):
            print(f"Training - {n + 1}/{num}")
            vgg16.model.fit(mydata.train_images_pgd_vgg16[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=1, batch_size=128)
        
        # save model
        vgg16.model.save(f'{model_save_to}/mnist_vgg16_model_with_pgd_vgg16.h5')
        print('mnist_vgg16_model_with_pgd_vgg16 Model Saved!')

        # save model
        vgg16.model.save_weights(f'{model_save_to}/mnist_vgg16_weights_with_pgd_vgg16.h5')
        print('mnist_vgg16_model_with_pgd_vgg16 Weights Saved!')

    if run_eval_clean:
        test_loss, test_accuracy = vgg16.model.evaluate(x=mydata.test_images, y=mydata.test_labels, batch_size=128)
        print(f'mnist_vgg16_model - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
        # mnist_vgg16_model - Test loss: 0.04019283875823021. Test Accuracy: 0.9871000051498413

    if run_eval_fgsm:
        test_loss, test_accuracy = vgg16.model.evaluate(x=mydata.test_images_fgsm_vgg16, y=mydata.test_labels, batch_size=128)
        print(f'mnist_vgg16_model on self-generated FGSM attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
        # mnist_vgg16_model on self-generated FGSM attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213
   
    if run_eval_pgd:
        test_loss, test_accuracy = vgg16.model.evaluate(x=mydata.test_images_pgd_vgg16, y=mydata.test_labels, batch_size=128)
        print(f'mnist_vgg16_model on self-generated PGD attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
        # mnist_vgg16_model on self-generated PGD attrack data - Test loss: 5.915209770202637. Test Accuracy: 0.08429999649524689

    return vgg16

def effnetB5_training(mydata, preload_model=None, model_name_set_to=None, model_save_to='saved_models_no_given_dir', generate_images_dir_path=None, train_model_clean=False, run_eval_clean=False, run_eval_fgsm=False, run_eval_pgd=False, 
                      generate_images_fgsm=False, generate_images_pgd=False, train_model_fgsm=False, train_model_pgd=False):
    effnetB5 = MyModel(10)
    effnetB5.preload_effnetb5()
    effnetB5.set_model_name(model_name_set_to)

    if preload_model:
        # load trained effnetB5 with mnist
        effnetB5.model=load_model(preload_model)
        effnetB5.model.summary()
        print(f'{preload_model} Model Loaded!')

    if train_model_clean:
        # train preload effnetB5 with mnist
        batch_size = 1000
        num = int(len(mydata.train_labels) / batch_size)
        for n in range(num):
            print(f"Training - {n + 1}/{num}")
            effnetB5.model.fit(mydata.train_images[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=8, batch_size=128)
        
        # save model
        effnetB5.model.save(f'{model_save_to}/mnist_effnetB5_model.h5')
        print('mnist_effnetB5_model Model Saved!')

        # save model
        effnetB5.model.save_weights(f'{model_save_to}/mnist_effnetB5_weights.h5')
        print('mnist_effnetB5_model Weights Saved!')

    if generate_images_fgsm:
        mydata.train_images_fgsm_effnetB5 = generate_adversarial_images_with_fgsm(effnetB5.model, mydata.train_images, "train_images_fgsm_effnetB5.npy", dir_path=generate_images_dir_path)
        mydata.test_images_fgsm_effnetB5 = generate_adversarial_images_with_fgsm(effnetB5.model, mydata.test_images, "test_images_fgsm_effnetB5.npy", dir_path=generate_images_dir_path)

    if train_model_fgsm:
        # train preload effnetB5 with mnist
        batch_size = 1000
        num = int(len(mydata.train_labels) / batch_size)
        for n in range(num):
            print(f"Training - {n + 1}/{num}")
            effnetB5.model.fit(mydata.train_images_fgsm_effnetB5[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=1, batch_size=128)
        
        # save model
        effnetB5.model.save(f'{model_save_to}/mnist_effnetB5_model_with_fgsm_effnetB5.h5')
        print('mnist_effnetB5_model_with_fgsm_effnetB5 Model Saved!')

        # save model
        effnetB5.model.save_weights(f'{model_save_to}/mnist_effnetB5_weights_with_fgsm_effnetB5.h5')
        print('mnist_effnetB5_model_with_fgsm_effnetB5 Weights Saved!')

    if generate_images_pgd:
        mydata.train_images_pgd_effnetB5 = generate_adversarial_images_with_pgd(effnetB5.model, mydata.train_images, "train_images_pgd_effnetB5.npy", dir_path=generate_images_dir_path)
        mydata.test_images_pgd_effnetB5 = generate_adversarial_images_with_pgd(effnetB5.model, mydata.test_images, "test_images_pgd_effnetB5.npy", dir_path=generate_images_dir_path)

    if train_model_pgd:
        # train preload effnetB5 with mnist
        batch_size = 1000
        num = int(len(mydata.train_labels) / batch_size)
        for n in range(num):
            print(f"Training - {n + 1}/{num}")
            effnetB5.model.fit(mydata.train_images_pgd_effnetB5[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=1, batch_size=128)
        
        # save model
        effnetB5.model.save(f'{model_save_to}/mnist_effnetB5_model_with_pgd_effnetB5.h5')
        print('mnist_effnetB5_model_with_pgd_effnetB5 Model Saved!')

        # save model
        effnetB5.model.save_weights(f'{model_save_to}/mnist_effnetB5_weights_with_pgd_effnetB5.h5')
        print('mnist_effnetB5_model_with_pgd_effnetB5 Weights Saved!')

    if run_eval_clean:
        test_loss, test_accuracy = effnetB5.model.evaluate(x=mydata.test_images, y=mydata.test_labels, batch_size=128)
        print(f'mnist_effnetB5_model - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
        # mnist_effnetB5_model - Test loss: 3.5259573459625244. Test Accuracy: 0.3474999964237213
    
    if run_eval_fgsm:  
        test_loss, test_accuracy = effnetB5.model.evaluate(x=mydata.test_images_fgsm_effnetB5, y=mydata.test_labels, batch_size=128)
        print(f'mnist_effnetB5_model on self-generated FGSM attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
        # mnist_effnetB5_model on self-generated FGSM attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213

    if run_eval_pgd:
        test_loss, test_accuracy = effnetB5.model.evaluate(x=mydata.test_images_pgd_effnetB5, y=mydata.test_labels, batch_size=128)
        print(f'mnist_effnetB5_model on self-generated PGD attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
        # mnist_effnetB5_model on self-generated PGD attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213
    
    return effnetB5


def resnet101_training(mydata, preload_model=None, model_name_set_to=None, model_save_to='saved_models_no_given_dir', generate_images_dir_path=None, train_model_clean=False, run_eval_clean=False, run_eval_fgsm=False, run_eval_pgd=False, 
                      generate_images_fgsm=False, generate_images_pgd=False, train_model_fgsm=False, train_model_pgd=False):
    resnet101 = MyModel(10)
    resnet101.preload_resnet101()
    resnet101.set_model_name(model_name_set_to)

    if preload_model:
        # load trained resnet101 with mnist
        resnet101.model=load_model(preload_model)
        resnet101.model.summary()
        print(f'{preload_model} Model Loaded!')

    if train_model_clean:
        # train preload resnet101 with mnist
        batch_size = 1000
        num = int(len(mydata.train_labels) / batch_size)
        for n in range(num):
            print(f"Training - {n + 1}/{num}")
            resnet101.model.fit(mydata.train_images[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=50, batch_size=128)
        
        # save model
        resnet101.model.save(f'{model_save_to}/mnist_resnet101_model.h5')
        print('mnist_resnet101_model Model Saved!')

        # save model
        resnet101.model.save_weights(f'{model_save_to}/mnist_resnet101_weights.h5')
        print('mnist_resnet101_model Weights Saved!')

    if generate_images_fgsm:
        mydata.train_images_fgsm_resnet101 = generate_adversarial_images_with_fgsm(resnet101.model, mydata.train_images, "train_images_fgsm_resnet101.npy", dir_path=generate_images_dir_path)
        mydata.test_images_fgsm_resnet101 = generate_adversarial_images_with_fgsm(resnet101.model, mydata.test_images, "test_images_fgsm_resnet101.npy", dir_path=generate_images_dir_path)

    if train_model_fgsm:
        # train preload resnet101 with mnist
        batch_size = 1000
        num = int(len(mydata.train_labels) / batch_size)
        for n in range(num):
            print(f"Training - {n + 1}/{num}")
            resnet101.model.fit(mydata.train_images_fgsm_resnet101[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=20, batch_size=128)
        
        # save model
        resnet101.model.save(f'{model_save_to}/mnist_resnet101_model_with_fgsm_resnet101.h5')
        print('mnist_resnet101_model_with_fgsm_resnet101 Model Saved!')

        # save model
        resnet101.model.save_weights(f'{model_save_to}/mnist_resnet101_weights_with_fgsm_resnet101.h5')
        print('mnist_resnet101_model_with_fgsm_resnet101 Weights Saved!')

    if generate_images_pgd:
        mydata.train_images_pgd_resnet101 = generate_adversarial_images_with_pgd(resnet101.model, mydata.train_images, "train_images_pgd_resnet101.npy", dir_path=generate_images_dir_path)
        mydata.test_images_pgd_resnet101 = generate_adversarial_images_with_pgd(resnet101.model, mydata.test_images, "test_images_pgd_resnet101.npy", dir_path=generate_images_dir_path)

    if train_model_pgd:
        # train preload resnet101 with mnist
        batch_size = 1000
        num = int(len(mydata.train_labels) / batch_size)
        for n in range(num):
            print(f"Training - {n + 1}/{num}")
            resnet101.model.fit(mydata.train_images_pgd_resnet101[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=1, batch_size=128)
        
        # save model
        resnet101.model.save(f'{model_save_to}/mnist_resnet101_model_with_pgd_resnet101.h5')
        print('mnist_resnet101_model_with_pgd_resnet101 Model Saved!')

        # save model
        resnet101.model.save_weights(f'{model_save_to}/mnist_resnet101_weights_with_pgd_resnet101.h5')
        print('mnist_resnet101_model_with_pgd_resnet101 Weights Saved!')

    if run_eval_clean:
        test_loss, test_accuracy = resnet101.model.evaluate(x=mydata.test_images, y=mydata.test_labels, batch_size=128)
        print(f'mnist_resnet101_model - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
        # mnist_resnet101_model - Test loss: 0.04019283875823021. Test Accuracy: 0.9871000051498413
    
    if run_eval_fgsm:  
        test_loss, test_accuracy = resnet101.model.evaluate(x=mydata.test_images_fgsm_resnet101, y=mydata.test_labels, batch_size=128)
        print(f'mnist_resnet101_model on self-generated FGSM attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
        # mnist_resnet101_model on self-generated FGSM attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213

    if run_eval_pgd:
        test_loss, test_accuracy = resnet101.model.evaluate(x=mydata.test_images_pgd_resnet101, y=mydata.test_labels, batch_size=128)
        print(f'mnist_resnet101_model on self-generated PGD attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
        # mnist_resnet101_model on self-generated PGD attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213
    
    return resnet101

def ensemble_training(mydata, target_model, pretrained_models=None, batch_size=1000, generate_images_dir_path=None):
    if not pretrained_models:
        return
    
    print("Running ensemble training!!!")
    model_cache_dir = f'saved_model_{target_model.name}_ensemble'
    for pretrained_model in pretrained_models:
        model_cache_dir += f'_{pretrained_model.name}'
    os.makedirs(model_cache_dir, exist_ok=True)
    source_models = [target_model] + pretrained_models
    num = int(len(mydata.train_labels) / batch_size)
    random_numbers = np.random.randint(0, len(source_models), size=num)
    for n in range(num):
        print(f"Training - {n + 1}/{num}")
        batch_train_images = mydata.train_images[batch_size * n: batch_size * (n + 1)]
        batch_train_labels = mydata.train_labels[batch_size * n: batch_size * (n + 1)]
        
        model_selected = source_models[random_numbers[n]]
        batch_adversarial_images = generate_adversarial_images_with_fgsm(model_selected.model, batch_train_images, f"train_images_fgsm_{batch_size * n}_{batch_size * (n + 1)}_{model_selected.name}.npy", dir_path=generate_images_dir_path)
        target_model.model.fit(batch_adversarial_images, batch_train_labels, epochs=8, batch_size=128)

        # save model
        target_model.model.save_weights(f'{model_cache_dir}/weights_{batch_size * (n + 1)}.h5')
        print(f'{model_cache_dir}/weights_{batch_size * (n + 1)}.h5 Weights Saved!')
    
    # save model
    target_model.model.save(f'{model_cache_dir}/mnist_{target_model.name}_model_ensemble_adversarial.h5')
    print(f'{model_cache_dir}/mnist_{target_model.name}_model_ensemble_adversarial.h5 Model Saved!')


def main():
    # # Main logic of the program
    # # mydata1 = MyDataset('mnist', (224, 224), 100, 10)
    # # mydata2 = MyDataset('mnist', (224, 224), 200, 50)
    # # mydata = MyDataset('mnist', (224, 224))

    # mydata = MyDataset('mnist', (224, 224), train_exmaple_counts=2000, test_example_counts=100)
    mydata = MyDataset('mnist', (224, 224))
    # mydata = MyDataset('preprocessed', (224, 224), processed_data_dir='dataset_mnist_train_60000_test_10000')
    # mydata = MyDataset('preprocessed', (224, 224), processed_data_dir='dataset_mnist_train_2000_test_100')
    
    # # effnetB5 = effnetB5_training(mydata, model_name_set_to='effnetB5', 
    # #                              model_save_to='saved_models_train_2000_test_100', 
    # #                              train_model_clean=True, run_eval_clean=True)
    # # effnetB5_fgsm_trained = effnetB5_training(mydata, model_name_set_to='effnetB5_fgsm_trained', 
    # #                                           preload_model='saved_models_train_2000_test_100/mnist_effnetB5_model.h5', 
    # #                                           model_save_to='saved_models_train_2000_test_100', 
    # #                                           generate_images_dir_path='dataset_mnist_train_2000_test_100',
    # #                                           generate_images_fgsm=True, train_model_fgsm=True)
    
    # # resnet50 = resnet50_training(mydata, model_name_set_to='resnet50', 
    # #                              model_save_to='saved_models_train_2000_test_100', 
    # #                              train_model_clean=True, run_eval_clean=True)
    # # resnet50_fgsm_trained = resnet50_training(mydata, model_name_set_to='resnet50_fgsm_trained', 
    # #                                           preload_model='saved_models_train_2000_test_100/mnist_resnet50_model.h5', 
    # #                                           model_save_to='saved_models_train_2000_test_100', 
    # #                                           generate_images_dir_path='dataset_mnist_train_2000_test_100',
    # #                                           generate_images_fgsm=True, train_model_fgsm=True)
    
    # # effnetB5 = effnetB5_training(mydata, model_name_set_to='effnetB5', preload_model='saved_models/mnist_effnetB5_model.h5', train_model_clean=True, run_eval_clean=True)
    # # effnetB5 = effnetB5_training(mydata, model_name_set_to='effnetB5', preload_model='saved_models/mnist_effnetB5_model.h5', train_model_clean=True, run_eval_clean=True)
    # # effnetB5_fgsm_trained = effnetB5_training(mydata, model_name_set_to='effnetB5_fgsm_trained', preload_model='saved_models/mnist_effnetB5_model.h5', generate_images_fgsm=True, train_model_fgsm=True)
    # # effnetB5_pgd_trained = effnetB5_training(mydata, preload_model='saved_models/mnist_effnetB5_model.h5', generate_images_pgd=True, train_model_pgd=True)
    # # vgg16 = vgg16_training(mydata, model_name_set_to='vgg16', model_save_to='saved_models_train_2000_test_100', train_model_clean=True, run_eval_clean=True)
    # # vgg16 = vgg16_training(mydata, model_name_set_to='vgg16',  preload_model='saved_models_train_2000_test_100/mnist_vgg16_model.h5')
    # # vgg16_fgsm_trained = vgg16_training(mydata, model_name_set_to='vgg16_fgsm_trained', preload_model='saved_models_train_2000_test_100/mnist_vgg16_model.h5', generate_images_dir_path='dataset_mnist_train_2000_test_100', generate_images_fgsm=True, train_model_fgsm=True)
    # # vgg16_pgd_trained = vgg16_training(mydata, preload_model='saved_models/mnist_vgg16_model.h5', generate_images_pgd=True, train_model_pgd=True)
    
    # # resnet50_fgsm_trained = resnet50_training(mydata, model_name_set_to='resnet50_fgsm_trained',  
    # #                              preload_model='saved_models_train_2000_test_100/mnist_resnet50_model_with_fgsm_resnet50.h5', 
    # #                              generate_images_fgsm=True,
    # #                              generate_images_dir_path="dataset_mnist_train_2000_test_100_resnet50_fgsm_trained")
    # # vgg16_fgsm_trained = vgg16_training(mydata, model_name_set_to='vgg16_fgsm_trained',  
    # #                               preload_model='saved_models_train_2000_test_100/mnist_vgg16_model_with_fgsm_vgg16.h5', 
    # #                               generate_images_fgsm=True,
    # #                               generate_images_dir_path="dataset_mnist_train_2000_test_100_vgg16_fgsm_trained")
    # # # vgg16 = vgg16_training(mydata, model_name_set_to='vgg16',  
    # # #                        preload_model='saved_models_train_2000_test_100/mnist_vgg16_model.h5')
    # # # pretrained_models = [vgg16_fgsm_trained, resnet50_fgsm_trained]
    # # # ensemble_training(mydata, vgg16, pretrained_models=pretrained_models, batch_size=200, generate_images_dir_path="dataset_mnist_vgg16_ensemble_training_vgg16_fgsm_trained_resnet50_fgsm_trained")
    # # vgg16_ensemble_trained = vgg16_training(mydata, model_name_set_to='vgg16_ensemble_trained',  
    # #                               preload_model='saved_model_vgg16_ensemble_vgg16_fgsm_trained_resnet50_fgsm_trained/mnist_vgg16_model_ensemble_adversarial.h5', 
    # #                               generate_images_fgsm=True,
    # #                               generate_images_dir_path="dataset_mnist_train_2000_test_100_vgg16_ensemble_trained")
    
    # # resnet101 = resnet101_training(mydata, model_name_set_to='resnet101',
    # #                                model_save_to='saved_models_train_2000_test_100', 
    # #                                train_model_clean=True, run_eval_clean=True)

    # resnet101 = resnet101_training(mydata, model_name_set_to='resnet101',
    #                                preload_model='saved_models_train_2000_test_100/mnist_resnet101_model.h5',
    #                                generate_images_fgsm=True, generate_images_pgd=True, 
    #                                generate_images_dir_path="dataset_mnist_train_2000_test_100")


    # test_loss, test_accuracy = vgg16.model.evaluate(x=mydata.test_images_fgsm_effnetB5, y=mydata.test_labels, batch_size=128)
    # print(f'mnist_vgg16_model on effnet generated FGSM attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
    # # mnist_vgg16_model on effnet generated FGSM attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213

    # test_loss, test_accuracy = vgg16.model.evaluate(x=mydata.test_images_pgd_effnetB5, y=mydata.test_labels, batch_size=128)
    # print(f'mnist_vgg16_model on effnet generated PGD attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
    # # mnist_vgg16_model on effnet generated PGD attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213

    # test_loss, test_accuracy = effnetB5.model.evaluate(x=mydata.test_images_fgsm_vgg16, y=mydata.test_labels, batch_size=128)
    # print(f'mnist_effnetB5_model on vgg generated FGSM attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
    # # mnist_effnetB5_model on vgg generated FGSM attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213

    # test_loss, test_accuracy = effnetB5.model.evaluate(x=mydata.test_images_pgd_vgg16, y=mydata.test_labels, batch_size=128)
    # print(f'mnist_effnetB5_model on vgg generated PGD attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
    # # mnist_effnetB5_model on vgg generated PGD attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213

    # print()


    # vgg16 = MyModel(10)
    # vgg16.preload_vgg16()

    # # load trained vgg16 with mnist
    # vgg16.model=load_model('saved_models/mnist_vgg16_model.h5')
    # vgg16.model.summary()
    # print('mnist_vgg16_model Model Loaded!')

    # # # train preload vgg16 with mnist
    # # batch_size = 1000
    # # num = int(len(mydata.train_labels) / batch_size)
    # # for n in range(num):
    # #     print(f"Training - {n + 1}/{num}")
    # #     vgg16.model.fit(mydata.train_images[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=1, batch_size=128)
    
    # # # save model
    # # vgg16.model.save('saved_models/mnist_vgg16_model.h5')
    # # print('mnist_vgg16_model Model Saved!')

    # # # save model
    # # vgg16.model.save_weights('saved_models/mnist_vgg16_weights.h5')
    # # print('mnist_vgg16_model Weights Saved!')
    
    
    # # # # vgg16.model.fit(mydata1.train_images, mydata1.train_labels, epochs=8, batch_size=32)
    # # # vgg16.model.fit(mydata2.train_images, mydata2.train_labels, epochs=8, batch_size=32)
    # # # vgg16.model.fit(mydata2.train_images, mydata2.train_labels, epochs=8, batch_size=32)
    # # test_loss, test_accuracy = vgg16.model.evaluate(x=mydata.test_images, y=mydata.test_labels, batch_size=128)
    # # print(f'mnist_vgg16_model - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
    # # # mnist_vgg16_model - Test loss: 0.04019283875823021. Test Accuracy: 0.9871000051498413

    # eps = 0.1
    # batch_size = 100
    # num = int(len(mydata.test_labels) / batch_size)
    # test_images_fgsm_mnist_vgg16_model = []
    # for n in range(num):
    #     print(f"Generating adversarial images - {n + 1}/{num}")
    #     generated_imgs = fast_gradient_method(vgg16.model, mydata.test_images[batch_size * n: batch_size * (n + 1)], eps, np.inf)
    #     if not len(test_images_fgsm_mnist_vgg16_model):
    #         test_images_fgsm_mnist_vgg16_model = generated_imgs
    #     else:
    #         test_images_fgsm_mnist_vgg16_model = np.vstack((test_images_fgsm_mnist_vgg16_model, generated_imgs))
    
    # dir_path = f"dataset_mnist_train_60000_test_10000"
    # os.makedirs(dir_path, exist_ok = True)

    # save_path = os.path.join(dir_path, "test_adversarial_images.npy")
    # print("Save to", save_path)
    # np.save(save_path, test_images_fgsm_mnist_vgg16_model)
    
    # test_loss, test_accuracy = vgg16.model.evaluate(x=test_images_fgsm_mnist_vgg16_model, y=mydata.test_labels, batch_size=128)
    # print(f'mnist_vgg16_model on self-generated FGSM attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
    # # mnist_vgg16_model on self-generated FGSM attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213


    print('Done')

# Check if the current module is the main module
if __name__ == "__main__":
    # Call the main function 
    main()

































