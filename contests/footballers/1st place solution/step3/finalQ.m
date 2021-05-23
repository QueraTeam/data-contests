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
Players{ix};
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
Players{idxx};
Players{idx};

NM=nshoots-NG;
%%
load finalmodel
load traindata
Y=SAE(traindata');
Y=Y(2,:)';
PrG=zeros(1,numel(Players));
PrM=zeros(1,numel(Players));
for j=1:numel(Players)
    G=[];
    M=[];
    for i=2:size(tt,1)
        if isequal(tt{i,2},Players{j})
            if isequal(tt{i,end},'G')
                PrG(j)=PrG(j)+ Y(i-1);
            else
                PrM(j)=PrM(j)+ Y(i-1);
            end
        end
    end

end

%%
Score = (100*NG ./ PrG  - NM .* PrM) ./ nshoots