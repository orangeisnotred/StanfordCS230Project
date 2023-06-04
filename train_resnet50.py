import os
from tensorflow.keras.models import load_model
from run_models import MyDataset, MyModel, generate_adversarial_images_with_fgsm, generate_adversarial_images_with_pgd

os.environ["TF_METAL_ENABLED"] = "1"

def resnet50_training(mydata, preload_model=None, model_name_set_to=None, model_save_to='saved_models_no_given_dir', generate_images_dir_path=None, train_model_clean=False, run_eval_clean=False, run_eval_fgsm=False, run_eval_pgd=False, 
                      generate_images_fgsm=False, generate_images_pgd=False, train_model_fgsm=False, train_model_pgd=False):
    resnet50 = MyModel(10)
    resnet50.preload_resnet50()
    resnet50.set_model_name(model_name_set_to)

    if preload_model:
        # load trained resnet50 with mnist
        resnet50.model=load_model(preload_model)
        resnet50.model.summary()
        print(f'{preload_model} Model Loaded!')

    if train_model_clean:
        # # train preload resnet50 with mnist
        # batch_size = 1000
        # num = int(len(mydata.train_labels) / batch_size)
        # for n in range(num):
        #     print(f"Training - {n + 1}/{num}")
        #     resnet50.model.fit(mydata.train_images[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=50, batch_size=128)

        num_epochs = 100
        batch_size = 64
        train_data, val_data, train_labels, val_labels, shuffled_indices = mydata.random_split_train_validation_set(mydata.train_images, mydata.train_labels, batch_size)
        # resnet50.model.fit(
        #     train_iterator,
        #     epochs=num_epochs,
        #     steps_per_epoch=len(train_dataset) // batch_size,
        #     validation_data=val_iterator,
        #     validation_steps=len(val_dataset) // batch_size,
        #     callbacks=[resnet50.early_stopping] 
        # )
        # resnet50.model.fit(
        #     train_data,
        #     train_labels,
        #     batch_size=64,
        #     epochs=100,
        #     validation_data=(val_data, val_labels),
        #     callbacks=[resnet50.early_stopping],
            
        # )
        
        # # save model
        # resnet50.model.save(f'{model_save_to}/mnist_resnet50_model.h5')
        # print('mnist_resnet50_model Model Saved!')

        # # save model
        # resnet50.model.save_weights(f'{model_save_to}/mnist_resnet50_weights.h5')
        # print('mnist_resnet50_model Weights Saved!')

    if generate_images_fgsm:
        mydata.train_images_fgsm_resnet50 = generate_adversarial_images_with_fgsm(resnet50.model, mydata.train_images, "train_images_fgsm_resnet50.npy", dir_path=generate_images_dir_path)
        mydata.test_images_fgsm_resnet50 = generate_adversarial_images_with_fgsm(resnet50.model, mydata.test_images, "test_images_fgsm_resnet50.npy", dir_path=generate_images_dir_path)

    if train_model_fgsm:
        # train preload resnet50 with mnist
        batch_size = 1000
        num = int(len(mydata.train_labels) / batch_size)
        for n in range(num):
            print(f"Training - {n + 1}/{num}")
            resnet50.model.fit(mydata.train_images_fgsm_resnet50[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=20, batch_size=128)
        
        # save model
        resnet50.model.save(f'{model_save_to}/mnist_resnet50_model_with_fgsm_resnet50.h5')
        print('mnist_resnet50_model_with_fgsm_resnet50 Model Saved!')

        # save model
        resnet50.model.save_weights(f'{model_save_to}/mnist_resnet50_weights_with_fgsm_resnet50.h5')
        print('mnist_resnet50_model_with_fgsm_resnet50 Weights Saved!')

    if generate_images_pgd:
        mydata.train_images_pgd_resnet50 = generate_adversarial_images_with_pgd(resnet50.model, mydata.train_images, "train_images_pgd_resnet50.npy", dir_path=generate_images_dir_path)
        mydata.test_images_pgd_resnet50 = generate_adversarial_images_with_pgd(resnet50.model, mydata.test_images, "test_images_pgd_resnet50.npy", dir_path=generate_images_dir_path)

    if train_model_pgd:
        # train preload resnet50 with mnist
        batch_size = 1000
        num = int(len(mydata.train_labels) / batch_size)
        for n in range(num):
            print(f"Training - {n + 1}/{num}")
            resnet50.model.fit(mydata.train_images_pgd_resnet50[batch_size * n: batch_size * (n + 1)], mydata.train_labels[batch_size * n: batch_size * (n + 1)], epochs=1, batch_size=128)
        
        # save model
        resnet50.model.save(f'{model_save_to}/mnist_resnet50_model_with_pgd_resnet50.h5')
        print('mnist_resnet50_model_with_pgd_resnet50 Model Saved!')

        # save model
        resnet50.model.save_weights(f'{model_save_to}/mnist_resnet50_weights_with_pgd_resnet50.h5')
        print('mnist_resnet50_model_with_pgd_resnet50 Weights Saved!')

    if run_eval_clean:
        test_loss, test_accuracy = resnet50.model.evaluate(x=mydata.test_images, y=mydata.test_labels, batch_size=128)
        print(f'mnist_resnet50_model - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
        # mnist_resnet50_model - Test loss: 0.04019283875823021. Test Accuracy: 0.9871000051498413
    
    if run_eval_fgsm:  
        test_loss, test_accuracy = resnet50.model.evaluate(x=mydata.test_images_fgsm_resnet50, y=mydata.test_labels, batch_size=128)
        print(f'mnist_resnet50_model on self-generated FGSM attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
        # mnist_resnet50_model on self-generated FGSM attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213

    if run_eval_pgd:
        test_loss, test_accuracy = resnet50.model.evaluate(x=mydata.test_images_pgd_resnet50, y=mydata.test_labels, batch_size=128)
        print(f'mnist_resnet50_model on self-generated PGD attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
        # mnist_resnet50_model on self-generated PGD attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213
    
    return resnet50

def main():

    mydata = MyDataset('preprocessed', (224, 224), processed_data_dir='dataset_mnist_train_60000_test_10000')
    resnet50 = resnet50_training(mydata, model_name_set_to='resnet50', 
                                 model_save_to='saved_models_train_60000_test_10000', 
                                 train_model_clean=True, run_eval_clean=True)
    resnet50_fgsm_trained = resnet50_training(mydata, model_name_set_to='resnet50_fgsm_trained', 
                                              preload_model='saved_models_train_60000_test_10000/mnist_resnet50_model.h5', 
                                              model_save_to='saved_models_train_60000_test_10000', 
                                              generate_images_dir_path='dataset_mnist_train_60000_test_10000',
                                              generate_images_fgsm=True, train_model_fgsm=True)
    

    # resnet50_fgsm_trained = resnet50_training(mydata, model_name_set_to='resnet50_fgsm_trained',  
    #                              preload_model='saved_models_train_2000_test_100/mnist_resnet50_model_with_fgsm_resnet50.h5', 
    #                              generate_images_fgsm=True,
    #                              generate_images_dir_path="dataset_mnist_train_2000_test_100_resnet50_fgsm_trained")
    # vgg16_fgsm_trained = vgg16_training(mydata, model_name_set_to='vgg16_fgsm_trained',  
    #                               preload_model='saved_models_train_2000_test_100/mnist_vgg16_model_with_fgsm_vgg16.h5', 
    #                               generate_images_fgsm=True,
    #                               generate_images_dir_path="dataset_mnist_train_2000_test_100_vgg16_fgsm_trained")
    # # vgg16 = vgg16_training(mydata, model_name_set_to='vgg16',  
    # #                        preload_model='saved_models_train_2000_test_100/mnist_vgg16_model.h5')
    # # pretrained_models = [vgg16_fgsm_trained, resnet50_fgsm_trained]
    # # ensemble_training(mydata, vgg16, pretrained_models=pretrained_models, batch_size=200, generate_images_dir_path="dataset_mnist_vgg16_ensemble_training_vgg16_fgsm_trained_resnet50_fgsm_trained")
    # vgg16_ensemble_trained = vgg16_training(mydata, model_name_set_to='vgg16_ensemble_trained',  
    #                               preload_model='saved_model_vgg16_ensemble_vgg16_fgsm_trained_resnet50_fgsm_trained/mnist_vgg16_model_ensemble_adversarial.h5', 
    #                               generate_images_fgsm=True,
    #                               generate_images_dir_path="dataset_mnist_train_2000_test_100_vgg16_ensemble_trained")
    
    # resnet101 = resnet101_training(mydata, model_name_set_to='resnet101',
    #                                model_save_to='saved_models_train_2000_test_100', 
    #                                train_model_clean=True, run_eval_clean=True)

    resnet101 = resnet101_training(mydata, model_name_set_to='resnet101',
                                   preload_model='saved_models_train_2000_test_100/mnist_resnet101_model.h5',
                                   generate_images_fgsm=True, generate_images_pgd=True, 
                                   generate_images_dir_path="dataset_mnist_train_2000_test_100")


    test_loss, test_accuracy = vgg16.model.evaluate(x=mydata.test_images_fgsm_effnetB5, y=mydata.test_labels, batch_size=128)
    print(f'mnist_vgg16_model on effnet generated FGSM attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
    # mnist_vgg16_model on effnet generated FGSM attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213

    test_loss, test_accuracy = vgg16.model.evaluate(x=mydata.test_images_pgd_effnetB5, y=mydata.test_labels, batch_size=128)
    print(f'mnist_vgg16_model on effnet generated PGD attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
    # mnist_vgg16_model on effnet generated PGD attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213

    test_loss, test_accuracy = effnetB5.model.evaluate(x=mydata.test_images_fgsm_vgg16, y=mydata.test_labels, batch_size=128)
    print(f'mnist_effnetB5_model on vgg generated FGSM attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
    # mnist_effnetB5_model on vgg generated FGSM attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213

    test_loss, test_accuracy = effnetB5.model.evaluate(x=mydata.test_images_pgd_vgg16, y=mydata.test_labels, batch_size=128)
    print(f'mnist_effnetB5_model on vgg generated PGD attrack data - Test loss: {test_loss}. Test Accuracy: {test_accuracy}')
    # mnist_effnetB5_model on vgg generated PGD attrack data - Test loss: 1.5758944749832153. Test Accuracy: 0.49619999527931213

    print()


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


    print()

# Check if the current module is the main module
if __name__ == "__main__":
    # Call the main function 
    main()

































