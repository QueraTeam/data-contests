%% This code provides the final output as a .xlsx file.
% Also it calculates the rqd values for further steps.
% To execute the code, make sure you have the "test-rqd" folder here
% also, deep learning toolboxes should be installed. 
clc
clear
close all
%% load the trained networks and images:
global net net2 net3
load WoodNet.mat
load RockNet.mat
load BoxNet.mat
Npic=[9 18 21];
pic={'M3-BH3299','M3-BH3300','M3-BH3301'};
%% let's do it:
k=1;
for p=1:3
    for i=1:Npic(p)
        str=[pic{p},'-',num2str(i),'.jpg'];
        I=imread(['test-rqd/',str]);
        Rows=segmentRows(I); % dividing the image into 5 rows
        Runs=row2run(Rows);  % finding the Runs in the rows
        for j=1:numel(Runs)
            name=[str(1:end-4),'-',num2str(j)];
            [out,rqd]=RQDclass(Runs(j),0); % RQD calculation
            disp([name,',',num2str(rqd),',',num2str(out)])
            X{k,1}=name;
            X{k,2}=num2str(out);
            k=k+1;
        end
    end
end
%% save the result:
xlswrite('output.xlsx',X)
