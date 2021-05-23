clc
clear
close all
%%
load traindata
load testdata
L=zeros(2,numel(labels));
L(1,labels==0)=1;
L(2,labels==1)=1;

%%
net=patternnet(10);
net=train(net,traindata',labels');
Y=net(testdata');
Y=Y';