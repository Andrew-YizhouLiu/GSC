function [y1,y2] = gsc_fit(y,x,weight,coef,state,N)
[x1,x2] = size(y);
temp = zeros(x1,x2);
for i = 1:N
    temp(:,i) = y(:,i) - squeeze(x(:,:,i))*coef;
end
y1 = temp(:,state);
temp(:,state) = [];
y2 = temp*weight(:,state);
end