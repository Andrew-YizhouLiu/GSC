function weight = calculate_weight(y,x,coef,N,init,Aeq,beq,lb,ub,options)
weight = zeros(N-1,N);
parfor j=1:N
    a = y(:,j);
    b = x(:,:,j);
    c = y;
    d = x;
    c(:,j) = [];
    d(:,:,j) = [];
    fun = @(phi)costfun(a,b,phi,c,d,coef);
    weight(:,j) = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
end


end