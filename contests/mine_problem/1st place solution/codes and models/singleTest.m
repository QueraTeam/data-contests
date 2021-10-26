clc
clear
close all
%%
global net net2 net3
load WoodNet.mat
load RockNet.mat
load BoxNet.mat
str='test-rqd/M3-BH3301-5.jpg';
I=imread(str);
figure, imshow(I)
Rows=segmentRows(I);
Runs=row2run(Rows);
idx=find(str=='/');
for j=1:numel(Runs)
    name=[str(idx(end)+1:end-4),'-',num2str(j)];
    [out,rqd]=RQDclass(Runs(j),1);
    title(['Run #',num2str(j),', RQD=',num2str(out)])
    disp([name,',',num2str(rqd),',',num2str(out)])
end