# CIFAR-10 Classification with the Matlab Deep Learning Toolbox
This is a matlab implementation of a tutorial in pytorch in the link below:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

The classification problem is implemented using the MATLAB deep learning toolbox. This toolbox provides users with visual design and training frameworks that simplify the process of using deep neural networks and minimize the required amount of coding. 
### Designing the Network Architecture
The deep learning toolbox offers a visual neural network design environment that can be launched by typing `deepNetworkDesigner` in the command window.
Depending on the type of network, the user can choose between different layer types, activation functions, and loss functions. The architecture can be assembled by drag&dropping the desired layers in the environment and connecting them together. Here we use the network structure used in the original tutorial. A view of the deep network designer environment is shown in the figure below.  

![deepdesigner](https://user-images.githubusercontent.com/57267379/84435245-58458400-abe6-11ea-81ce-1ebd16d5db77.jpg)

Layer parameters, such as stride and padding can be adjusted the properties tab for the corresponding layer. Note that in the original problem in no padding and L2 regularization were used, therefore we need to adjust these properties in for all of the layers. Hence, we choose [0,0,0,0] as the padding value (not to be mistaken with zero padding, the values inside the bracket represent the number of rows/columns to be added to sides of the convolutional layer) and also set the L2 regularization value to zero for all of the layers. We could also have left the L2 regularization value of the layers unchanged and set its value when choosing the training options. We also choose rescale-symmetric for the imageinputlayer to normalize the data in the [-1,1] range. All other parameters are left unchanged. Finally, a softmax and a classifier layer are added in the end of the model. The classifier layer applies the cross entropy loss function on the outputs of its previous layer,i.e., softmax. After designing the the network, we can click on analyze to check for structural flaws in the network. 

![analysis](https://user-images.githubusercontent.com/57267379/84435243-57aced80-abe6-11ea-906c-c6cfce7eb0b8.jpg)

As can be seen in Figure 8, the model is compiled without any warnings or errors. Now we can use the CIFAR-10 dataset to train this network. 
### Importing the CIFAR-10 Dataset
By typing the following code in the command window, we can download the CIFAR-10 dataset and load it in the matlab workspace:

    cifar10Data = tempdir; 
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz'; 
    helperCIFAR10Data.download(url,cifar10Data); 
    [trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load(cifar10Data);
    
Similar to the pytorch implementation we can visualize the data using the imshow command. Below is a sample image from the dataset:

    random=randi([1 50000],1,4)

    for i=1:4
    subplot(1,4,i), imshow(:,:,:,random(i))
    end
    
![Sample Images](https://user-images.githubusercontent.com/57267379/84435246-58458400-abe6-11ea-8c22-2ca7f3580d01.jpg)

### Training the Network

After loading both the network and the dataset in the workplace, we can adjust the desired set of options and train the network. We choose stochastic gradient descent with a batch size of 4 and train the network for 10 epochs. Please refer to the provided MATLAB code for other training options. Finally, the training can be initiated using the following command:
    net = trainNetwork(trainingImages, trainingLabels,layers,opts);
The following figure shows the learning progression throughout all 10 epochs.

![Training_Progress](https://user-images.githubusercontent.com/57267379/84435247-58de1a80-abe6-11ea-99dd-fc1059effe03.png)

Note that high-frequency fluctuations in accuracy and loss value are caused by choosing a far too small batch size. We can address this problem by choosing a larger batch size. Overall, network accuracy has improvement as predicted. Now that the network parameters have been updated, we can save the model by the `save net` command.

