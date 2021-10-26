function run=row2run(Rows)
    global net net3
    for i=1:5
        R=imresize(Rows{i},[300,1000]);
        C=semanticseg(R,net);
%         figure
%         subplot(2,1,1), imshow(R)
        C=C=='wood';
%         ids=find(sum(C)<0.1*size(C,2));
%         C(:,ids)=0;
        bwRows{i}=C;
        bwRows{i}=medfilt2(C,[50,5]);
%         subplot(2,1,2), imshow(bwRows{i})
        Cs=bwboundaries(bwRows{i});
        kk=1;
        newCs={};
        for k=1:numel(Cs)
            
            if numel(Cs{k})>400
                newCs{kk}=Cs{k};
                kk=kk+1;
            end
        end
        Cs=newCs;
        if isempty(Cs)
            segment{i}{1}=R; 
        elseif numel(Cs)==1
            segment{i}{1}=R(:,1:min(Cs{1}(:,2)),:);
            segment{i}{2}=R(:,max(Cs{1}(:,2)):end,:);
        elseif numel(Cs)==2
            segment{i}{1}=R(:,1:min(Cs{1}(:,2)),:);
            segment{i}{2}=R(:,max(Cs{1}(:,2)):min(Cs{2}(:,2)),:);
            segment{i}{3}=R(:,max(Cs{2}(:,2)):end,:);
        else
            segment{i}{1}=R(:,1:min(Cs{1}(:,2)),:);
            segment{i}{2}=R(:,max(Cs{1}(:,2)):min(Cs{2}(:,2)),:);
            segment{i}{3}=R(:,max(Cs{2}(:,2)):min(Cs{3}(:,2)),:);
            segment{i}{4}=R(:,max(Cs{3}(:,2)):end,:);
        end   
    end
    r=1;
    run{r}=[];
    for i=1:5
        run{r}=[run{r}, segment{i}{1}];
        if numel(segment{i})==2
            r=r+1;
            run{r}=[segment{i}{2}];
        elseif numel(segment{i})==3
            r=r+1;
            run{r}=[segment{i}{2}];
            r=r+1;
            run{r}=[segment{i}{3}];
        elseif numel(segment{i})==4
            r=r+1;
            run{r}=[segment{i}{2}];
            r=r+1;
            run{r}=[segment{i}{3}];
            r=r+1;
            run{r}=[segment{i}{4}];
        end
    end
    if size(run{end},2)<100
        run=run(1:end-1);
    end
    
    I=imresize(run{end},[224 224]);

    boxStat=predict(net3,I);
    if boxStat(1)>0.99 % So it's certainly empty. remove it.
        run=run(1:end-1);
    end
end