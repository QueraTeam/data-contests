function out=objfcn(L,S)
    L=sort(L);
    D=diff([L,numel(S)]);
    if sum(D<50)>0 || L(1)<10 || L(end)> numel(S)-10
        out=0.001;
        return;
    end
    sm=0;
    for i=1:numel(L)
        sm=sm+ sum(S(L(i)-5:L(i)+5));
    end
    out= (sum(D)*sm*sum(S(L)))/(10+std(D));
end