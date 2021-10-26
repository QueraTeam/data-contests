clc
clear
close all
%% load data
path_train= fullfile('Data');
x_train= imageDatastore(path_train, ...
    'IncludeSubfolders',true,'FileExtensions', ...
    '.jpg','LabelSource','foldernames');
num_label=countEachLabel(x_train)
%% load network
net=googlenet;
%% change last layer
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
featurelayer='loss3-classifier';
numClasses = numel(categories(x_train.Labels));
new_fc = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,featurelayer,new_fc);
classes={'empty','full'};
endLayer=classificationLayer('Classes',classes, 'name', 'last');
lgraph = replaceLayer(lgraph,'output',endLayer);
miniBatchSize = 10;
valFrequency = floor(numel(x_train.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');
net3 = trainNetwork(x_train,lgraph,options);
%% predict
[YPred,probs] = classify(net3,x_train);
accuracy = mean(YPred == x_train.Labels)
save('BoxNet.mat','net3');