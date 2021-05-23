clc
clear
close all
%%
[~,~,tt]=xlsread('test.xlsx');
testdata=[];
for j=[1,3:9]
    for i=2:size(tt,1)
        if ~isstr(tt{i,j})
            rows(i,1)= tt{i,j};
        else
            rows(i,1)= str2num(tt{i,j});
        end
    end
    testdata=[testdata , rows(2:end,1)];
end
BP=tt(2:end,2);
vec=onehotvec(BP);
[i,j]=find(isnan(testdata));
testdata(i,6)=1;
testdata=[testdata,vec];
%%
save('testdata.mat','testdata')