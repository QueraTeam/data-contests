function [out,rqd]=RQDclass(Runs,show)
global net2
for i = 1:numel(Runs)
    Runs{i}=[Runs{i} , zeros(300,1000,3)];
    C=semanticseg(Runs{i},net2);
    C=C=='rock';
    C=C(:,1:end-1000,:);
    T=size(C,1)*0.1;
    if show==1
        figure
        subplot(2,1,1);
        imshow(Runs{i}(:,1:end-1000,:))
        subplot(2,1,2);
        imshow(C) 
    end
    rqd(i)=sum(sum(C)>T)/(size(C,2)) *100+5; % 5 pixels for borders
    if rqd(i)<=25
        class=1;
    elseif rqd(i)<=50
        class=2;
    elseif rqd(i)<=75
        class=3;
    elseif rqd(i)<=90
        class=4;
    else
        class=5;
    end
    out(i)=class;
end

end