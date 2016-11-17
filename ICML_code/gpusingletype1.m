function x = gpusingletype1(x)
global usegpu gpusingletype;
if ~isempty(usegpu) && (numel(x) > 1)
    x = gpusingletype(x);
end
end