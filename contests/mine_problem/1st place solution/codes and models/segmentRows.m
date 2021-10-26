function Rows=segmentRows(I)
    r=round(size(I,1)*0.4);
    I=I(r:end,:,:);
    Ig=rgb2gray(I);
    [~,d2,~,~]=dwt2(Ig,'haar');
    D=d2>15;
    D=imfill(D,'holes');
    D=medfilt2(D);
    D=medfilt2(D);
    D=imresize(D,2);
    S=sum(D,2);
    S=smooth(S,10);
    S=medfilt1(S,5);
    %%
    npop=50;
    maxIter=10000;
    for i=1:npop
        pop{i}=randperm(numel(S),5);
    end
    for iter=1:maxIter
        for i=1:npop
            fitness(i)=objfcn(pop{i},S);
        end
        probs=fitness/sum(fitness);
        [~,idx]=sort(fitness);
        best=pop{idx(end)};
        for i=1:npop/2
            newpop{i}=pop{idx(end-i+1)};
        end
        for i=npop/2+1:npop
            pp=randsrc(1,2,[1:npop;probs]);
            p1=pop{pp(1)};
            p2=pop{pp(2)};
            child= p1;
            child(1:2)=p2(1:2);
            if rand>0.5
                r=randperm(5,1);
                child(r)=randperm(numel(S),1);
            end
            newpop{i}=child;
        end
        pop=newpop;
    end
    best=sort(best);
    %%
    boxW=10;
    Rows{1}=I(best(1)+boxW:best(2)-boxW,:,:);
    Rows{2}=I(best(2)+boxW:best(3)-boxW,:,:);
    Rows{3}=I(best(3)+boxW:best(4)-boxW,:,:);
    Rows{4}=I(best(4)+boxW:best(5)-boxW,:,:);
    Rows{5}=I(best(5)+boxW:end-boxW,:,:);
    
end