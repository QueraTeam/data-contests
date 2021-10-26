clc
clear
close all

%%
imgDir = 'Data/in';
imds = imageDatastore(imgDir);
I = readimage(imds,1);
imageSize = size(I);
classes = [
    "bg"
    "wood"
    ];
numClasses = numel(classes);
labelIDs{1} = 1;
labelIDs{2,1} = 2;
labelDir = fullfile('Data/out');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);
%% some info
tbl = countEachLabel(pxds);
frequency = tbl.PixelCount/sum(tbl.PixelCount);
bar(1:numel(classes),frequency);
xticks(1:numel(classes)) ;
xticklabels(tbl.Name);
xtickangle(45);
ylabel('Frequency');
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;
%% Network
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);
ds = combine(imds , pxds);
%% train 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'MaxEpochs',30, ...  
    'MiniBatchSize',2, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress');
%strat train
net = trainNetwork(ds,lgraph,options);
save('WoodNet.mat','net');