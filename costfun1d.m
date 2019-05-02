function out = costfun1d(y,d,phi,A,C,b)
out = (y-d*b-(A-C*b)*phi)'*(y-d*b-(A-C*b)*phi);
end

