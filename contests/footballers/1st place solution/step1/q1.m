clc
clear all
close all
%%
[~,~,tt]=xlsread('train.xlsx');
ids=[];
Players=unique((tt(2:end,2)));
for i=2:size(tt,1)
    if isequal(tt{i,end},'G')
        ids=[ids;i];
    end
end
NG=zeros(1,numel(Players));
for i=1:numel(ids)
    for j=1:numel(Players)
        if isequal(tt{ids(i),2},Players{j})
            NG(j)=NG(j)+1;
            
        end
    end
end
[~,ix]=max(NG);
Players{ix}
%%
nshoots=zeros(1,numel(Players));
for i=1:numel(Players)
    for j=1:size(tt,1)
        if isequal(Players{i},tt{j,2})
            nshoots(i)=nshoots(i)+1;
        end
    end
end
rate=NG./nshoots;
[valx,idxx]=max(rate);
[val,idx]=min(rate);
Players{idxx}
Players{idx}

NM=nshoots-NG