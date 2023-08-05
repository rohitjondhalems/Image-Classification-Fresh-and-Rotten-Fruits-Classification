# Image-Classification-Fresh-and-Rotten-Fruits-Classification
 to build a deep learning computer vision to model to classify fruits into categories Using Transfer Learning



The code is for building an image classification model using deep learning to classify fruits into six categories: fresh apples, fresh bananas, fresh oranges, rotten apples, rotten bananas, and rotten oranges. The code uses the VGG16 pre-trained model as a base model and adds additional layers on top of it for fine-tuning.

Here is a brief explanation of each part of the code:

1. Load ImageNet Base Model:
   - The code imports the necessary libraries, including `keras`.
   - It loads the VGG16 pre-trained model with weights from the "imagenet" dataset.
   - The input shape of the model is set to (224, 224, 3) since VGG16 expects images of size 224x224 with 3 channels (RGB).
   - The `include_top=False` argument means to exclude the fully connected layers at the top of the VGG16 model, as we will add our custom classification layers later.
   - The base model is frozen (non-trainable) so that only the new layers added will be trained.

2. Add Layers to Model:
   - A new input layer of shape (224, 224, 3) is created.
   - The output of the base model is passed through a Global Average Pooling 2D layer, which converts the 3D tensor to a 2D tensor by taking the average of all the values in each feature map.
   - A Dense layer with 6 units and a softmax activation function is added as the final classification layer to predict the probabilities of the six fruit categories.
   - The input and output layers are combined to create the final model.

3. Compile Model:
   - The model is compiled using the categorical cross-entropy loss function, which is suitable for multi-class classification tasks.
   - The Adam optimizer is used for training the model.
   - The accuracy metric is used to monitor the model's performance during training.

4. Augment the Data:
   - The code imports `ImageDataGenerator` from `tensorflow.keras.preprocessing.image`.
   - Image data augmentation is applied to the training data to increase the dataset's diversity and prevent overfitting. The augmentation includes random rotation, zooming, horizontal flipping, and vertical flipping.

5. Load Dataset:
   - The training and validation datasets are loaded using the `flow_from_directory` method of the `ImageDataGenerator`. The method reads images from the specified directories, performs data augmentation on the fly, and generates batches of training and validation data.

6. Train the Model:
   - The model is trained using the `fit` method.
   - Training data is provided from the `train_it` generator, and validation data is provided from the `valid_it` generator.
   - The `steps_per_epoch` and `validation_steps` are set to the number of samples divided by the batch size to specify how many batches should be processed in each epoch.

7. Unfreeze Model for Fine-Tuning:
   - After the initial training, the base model's layers are unfrozen to allow fine-tuning.
   - The model is recompiled with a low learning rate (0.00005) to avoid destroying the pre-trained weights.

8. Fine-Tune the Model:
   - The model is fine-tuned for additional epochs (45 more epochs) using the unfrozen base model.
   - The learning rate is low to prevent drastic updates to the pre-trained weights.

9. Evaluate the Model:
   - The model is evaluated on the validation dataset using the `evaluate` method, which returns the loss and accuracy.

The model achieves high accuracy on the validation set, indicating its success in classifying fresh and rotten fruits.
