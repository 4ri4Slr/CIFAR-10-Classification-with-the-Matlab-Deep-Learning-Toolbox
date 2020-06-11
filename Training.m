% Download and Load the CIFAR-10 Dataset

cifar10Data = tempdir;

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';

helperCIFAR10Data.download(url,cifar10Data);

[trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load(cifar10Data);

%Plotting random images from the dataset

random=randi([1 50000],1,4)

for i=1:4
subplot(1,4,i), imshow(:,:,:,random(i))
end

%% Load the network 

Layers.m 

%% Training Options
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 4, ...
    'Verbose', false, ...
"Plots","training-progress");

%% Training and saving the network

net = trainNetwork(trainingImages, trainingLabels,layers,opts);

save net
