clc
clear
close all
%%
load traindata2
load testdata
L=zeros(2,numel(labels));
L(1,labels==0)=1;
L(2,labels==1)=1;
%%
