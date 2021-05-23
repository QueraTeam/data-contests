clc
clear
close all
%%
[~,~,tt]=xlsread('train.xlsx');
traindata=[];
for j=[1,3:10]
    for i=2:size(tt,1)
        if ~isstr(tt{i,j})
            rows(i,1)= tt{i,j};
        else
            rows(i,1)= str2num(tt{i,j});
        end
    end
    traindata=[traindata , rows(2:end,1)];
end
labels=traindata(:,end);
traindata(:,end)=[];
BP=tt(2:end,2);
vec=onehotvec(BP);
[i,j]=find(isnan(traindata));
traindata(i,6)=1;
traindata=[traindata,vec];
%%
save('traindata.mat','traindata','labels')