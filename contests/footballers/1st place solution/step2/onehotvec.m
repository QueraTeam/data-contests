function out=onehotvec(vec)
    Uq={'L','R','H','O'};
    out=zeros(size(vec,1),numel(Uq));
    for i=1:numel(vec)
        for j=1:numel(Uq)
            if isequal(vec{i},Uq{j})
                out(i,j)=1;
            end
        end
    end
        
end