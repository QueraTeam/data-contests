clc
clear
close all
%%
load traindata2
load testdata
L=zeros(2,numel(labels));
L(1,labels==0)=1;
L(2,labels==1)=1;

Ftr=logical([1 1 1 1 1 1 1 0 1 1 1 1]);
%%
ae1=trainAutoencoder(traindata(:,Ftr)',8,...
        'L2WeightRegularization',0.01,...
        'SparsityRegularization',4,...
        'SparsityProportion',0.10);
feats=encode(ae1,traindata(:,Ftr)');
ae2=trainAutoencoder(feats,5,...
        'L2WeightRegularization',0.01,...
        'SparsityRegularization',4,...
        'SparsityProportion',0.10);
feats2=encode(ae2,feats);
SM=trainSoftmaxLayer(feats2,L);
SAE=stack(ae1,ae2,SM);
%%
SAE=train(SAE,traindata(:,Ftr)',L);
Y=SAE(testdata(:,Ftr)');
Y=Y(2,:)'