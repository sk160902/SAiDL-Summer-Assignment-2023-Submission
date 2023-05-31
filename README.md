# SAiDL-Summer-Assignment-2023-Submission

# Task 1: core ML:
A CNN model was developed with the following characteristics:
NN) for image classification. Here's a breakdown of the model:

Input Layer:

The input shape of the images is (32, 32, 3), representing 32x32 pixels with three color channels (RGB).
Convolutional Layers:

The first convolutional layer (Conv2D) has 32 filters with a kernel size of (3, 3).
The activation function used is ReLU, which introduces non-linearity to the model.
The output shape of this layer is determined by the number of filters and the spatial dimensions of the input.
Max Pooling Layers:

Following each convolutional layer, a max pooling layer (MaxPooling2D) is applied with a pool size of (2, 2).
Max pooling reduces the spatial dimensions of the input by selecting the maximum value within each pool window.
This helps in capturing the most salient features while reducing computational complexity.
Flattening Layer:

After the convolutional and max pooling layers, the feature maps are flattened (Flatten) into a 1D vector.
This prepares the data for the subsequent fully connected layers.
Fully Connected Layers:

The flattened features are connected to a dense layer (Dense) with 128 units.
The activation function used is ReLU, introducing non-linearity to the model.
This dense layer helps to capture higher-level representations of the input.
Output Layer:

The final dense layer has 100 units, corresponding to the 100 classes in the CIFAR-100 dataset.
There is no activation function specified for this layer, meaning it outputs raw logits.
The logits represent the model's predictions for each class.

Standard Softmax:
The standard softmax function is a deterministic operation that converts the logits (output of the neural network) into a probability distribution. It computes the exponentiated values of the logits and then normalizes them to sum up to 1. The softmax function is differentiable, and during training, it is typically used in conjunction with the cross-entropy loss.
Hence the accuracy being higher when compared to gumbel softmax

Gumbel-Softmax:
The Gumbel-Softmax is a relaxation technique that introduces stochasticity into the inference process. It is based on the Gumbel distribution, which is a continuous probability distribution that can be used to sample from a discrete distribution. The Gumbel-Softmax reparameterizes the logits by adding Gumbel noise and then applies the softmax function to obtain a differentiable approximation of the discrete distribution.

# bonus:

This code implements a convolutional neural network (CNN) with a Transformer-based architecture and a Gumbel-Softmax layer for the CIFAR-100 dataset. The CIFAR-100 dataset consists of 50,000 training images and 10,000 test images, each belonging to one of 100 classes.

The code preprocesses the data by scaling it between 0 and 1. It then defines a Gumbel-Softmax sampling function and a Transformer-based architecture. The architecture consists of convolutional layers, a Transformer layer, a Gumbel-Softmax layer, and an output layer.

The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss. It is trained on the training data for 10 epochs with a batch size of 64 and validated using the test data.

After training, the model is evaluated by predicting the labels for the test data. The predicted labels are compared with the true labels to calculate various evaluation metrics such as accuracy, precision, recall, F1 score, and confusion matrix. These metrics provide insights into the model's performance in classifying the CIFAR-100 images.


The Transformer architecture has lower performance compared to a traditional CNN architecture in image classification as the dataset primarily consists of local patterns and spatial hierarchies that can be effectively captured by CNNs.

## Task 2: CV

The CLIP (Contrastive Language-Image Pretraining) model is utilized to perform image-text matching and generate textual descriptions for images. It loads the CLIP model, encodes images and text using the model, calculates cosine similarity between them, and visualizes the results. It also demonstrates how to use the CLIP model for image classification by calculating the similarity between image features and class descriptions. The code showcases the ability of the CLIP model to understand the relationship between images and text, and its potential for tasks such as image retrieval, visual question answering, and text-based image generation.

A decoder is trained on top of it to produce a binary segmentation map however the U net model seems to give a better training accuracy.The U-Net architecture tends to provide better accuracy for image segmentation tasks compared to the simpler decoder model because it captures more detailed features through the encoder-decoder structure and skip connections, enabling better localization and segmentation of objects in the images.

# TASk 3: RL

A Decision Transformer model for sequence classification tasks is defined. The model takes an input sequence and applies an embedding layer and positional encoding. It then utilizes a Transformer encoder to capture dependencies between sequence elements. Finally, a decision MLP (Multi-Layer Perceptron) is applied to make predictions for each element in the sequence. The model is implemented using PyTorch.

A model instance is created with specified dimensions and parameters. Random input data is generated, and a forward pass is performed to obtain the model's output. The output shape represents the predictions for each element in the input sequence, with dimensions (batch_size, sequence_length, output_dim).

In the mentioned paper,the model was trained in the atari environment which gave a better performance when compared to the hopper environment as  atari involved discrete actions and visual input, allowing convolutional neural networks to effectively learn useful representations. On the other hand, the Hopper environment requires continuous control and has a more complex state space, making it challenging for traditional reinforcement learning algorithms. Second, the availability of pre-trained models and established benchmarks for Atari games enables better comparison and transfer learning. Lastly, the complexity and stability of the learning dynamics in the Atari games may make them more amenable to reinforcement learning algorithms, resulting in better results compared to the Hopper environment.





