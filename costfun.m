function out = costfun(y,d,phi,A,C,b)
n = size(C,3);
m = length(y);
if n==1
    out = (y-d*b-(A-b*C)*phi)'*(y-d*b-(A-b*C)*phi);
else
    DB = zeros(m,n);
    for i=1:n
        DB(:,i)=C(:,:,i)*b;
    end
    out = (y-d*b-(A-DB)*phi)'*(y-d*b-(A-DB)*phi);
end

        

    
    
