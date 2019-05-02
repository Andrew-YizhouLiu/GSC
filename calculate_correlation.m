function out = calculate_correlation(correlation,N)
out = zeros(N,1);
for i = 1:N
    y1 = correlation(:,1,i);
    y2 = correlation(:,2,i);
    temp = corrcoef(y1,y2);
    out(i,1) = temp(1,2);
end
