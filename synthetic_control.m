clc
clear 
cd 'C:\Users\Andre\Dropbox\Minimum wage and occupational mobility\synthetic control'

%{
data_teen = importdata('matlab_minwage_teen_cps.out');
data_45 = importdata('matlab_minwage_45_cps.out');
data_45 = importdata('matlab_minwage_45_cps_04.out');
data_45 = importdata('matlab_minwage_45_cps_07.out');
data_45 = importdata('matlab_minwage_45_cps_10.out');
data_45 = importdata('matlab_minwage_45_cps_11.out');
data_45 = importdata('matlab_minwage_45_cps_13.out');
data_45 = importdata('matlab_minwagewcontrol_45_cps_13.out');
data_45 = importdata('matlab_minwagewcontrol_45_empl_cps_13.out');
data_45 = importdata('matlab_minwagewcontrol_45_puiodp_cps_13.out');
data_30 = importdata('matlab_minwage_30_cps.out');
data_30 = importdata('matlab_minwage_30_cps_04.out');
data_30 = importdata('matlab_minwage_30_cps_07.out');
data_30 = importdata('matlab_minwage_30_cps_10.out');
data_30 = importdata('matlab_minwage_30_cps_11.out');
data_30 = importdata('matlab_minwage_30_cps_13.out');
data_30 = importdata('matlab_minwagewcontrol_30_cps_13.out');
data_30 = importdata('matlab_minwagewcontrol_30_empl_cps_13.out');
data_30 = importdata('matlab_minwagewcontrol_30_puiodp_cps_13.out');
data_hs = importdata('matlab_minwage_hs_cps_12.out');
data_cl = importdata('matlab_minwage_cl_cps_13.out');
data_cl = importdata('matlab_minwagewcontrol_cl_cps_13.out');
data_cl = importdata('matlab_minwagewcontrol_cl_empl_cps_13.out');
data_cl = importdata('matlab_minwagewcontrol_cl_puiodp_cps_13.out');
data_lhs = importdata('matlab_minwage_lhs_cps_13.out');
data_lhs = importdata('matlab_minwagewcontrol_lesshs_cps_13.out');
data_lhs = importdata('matlab_minwagewcontrol_lesshs_empl_cps_13.out');
data_lhs = importdata('matlab_minwagewcontrol_lesshs_puiodp_cps_13.out');
data= data_teen.data;
data = data_45.data;
data = data_30.data;
data = data_hs.data;
data = data_cl.data;
data = data_lhs.data;
%}
data_hs_30 = importdata('matlab_minwage_hs_30.out');
data = data_hs_30.data;

data_lowwage = importdata('matlab_minwage_lowwage.out');
data = data_lowwage.data;

data_30 = importdata('matlab_minwage_30.out');
data = data_30.data;

data_college_30 = importdata('matlab_minwage_college_30.out');
data = data_college_30.data;

data_highwage = importdata('matlab_minwage_highwage.out');
data = data_highwage.data;
%{
outcome{1} = [1,1,1];
disp(outcome{1}(1,1));
%}

%% First assign data to vectors

% This step retrieves relavant data, which is all of them :)
state = data(:,1);
outcome = data(:,2);
minwage = data(:,3);
manu_share = data(:,4);
re_share = data(:,5);

% Reshape the data for easy implementation later
outcome = reshape(outcome,[],51);
minwage = reshape(minwage,[],51);
manu_share = reshape(manu_share,[],51);
re_share = reshape(re_share,[],51);
[T,N] = size(minwage);
treatment = zeros(T,3,N);
treatment(:,1,:) = minwage;
treatment(:,2,:) = manu_share;
treatment(:,3,:) = re_share;

%{
for i = 1:56
    for j = 1:length(data)
        if data(j,1) == i
            outcome{i} = cat(1,outcome{i},data(j,2));  % using the true switch here
            treatment{i} = cat(1,treatment{i},data(j,[3:5]));
        end
    end
end

outcome = outcome(~cellfun('isempty',outcome)); %delete empty entries
treatment = treatment(~cellfun('isempty',treatment));
N = length(outcome);
T = length(outcome{1});
%}

%{
for i = 1:length(outcome)
    for j = 1:length(outcome{i})
        %test = ismember(time{i}(j),time{26});
        if test == 0
            outcome{i}(j) = 0;
            treatment{i}(j) = 0;
            %time{i}(j) = 0;
            %state{i}(j) = 0;
        end
    end
end
            
% Again drop the element which is 0
for i = 1:length(outcome)
    outcome{i} = nonzeros(outcome{i});
    treatment{i} = nonzeros(treatment{i});
    time{i} = nonzeros(time{i});
    state{i} = nonzeros(state{i});
end

dim = zeros(length(outcome),1);
for i = 1:length(dim)
    dim(i) = length(outcome{i});
end
[val1,idx1] = max(dim);
[val2,idx2] = min(dim);

while val1 > val2
    for i = 1:length(outcome)
        for j = 1:length(outcome{i})
            test = ismember(time{i}(j),time{idx2});
            if test == 0
                outcome{i}(j) = 0;
                treatment{i}(j) = 0;
                time{i}(j) = 0;
                state{i}(j) = 0;
            end
        end
    end
            
    % Again drop the element which is 0
    for i = 1:length(outcome)
        outcome{i} = nonzeros(outcome{i});
        treatment{i} = nonzeros(treatment{i});
        time{i} = nonzeros(time{i});
        state{i} = nonzeros(state{i});
    end
    
    % Delete empty cells
    outcome = outcome(~cellfun('isempty',outcome)); %delete empty entries
    treatment = treatment(~cellfun('isempty',treatment));
    time = time(~cellfun('isempty',time));
    state = state(~cellfun('isempty',state));

    dim = zeros(length(outcome),1);
    for i = 1:length(dim)
        dim(i) = length(outcome{i});
    end
    [val1,idx1] = max(dim);
    [val2,idx2] = min(dim);
end
%}
%{
% We now do a trial run
y = outcome{1};
d = treatment{1};
b = -0.02;
A = outcome{1};
C = treatment{1};
for i = 2:N
    A = cat(2,A,outcome{i});
    C = cat(2,C,treatment{i});
end
phi = zeros(50,1);
A(:,1) = [];
C(:,1) = [];
fun = @(phi)costfun(y,d,phi,A,C,b);
lb = zeros(length(phi),1);
ub = ones(length(phi),1);
Aeq = ones(1,length(phi));
beq = 1;
trial = fmincon(fun,phi,[],[],Aeq,beq,lb,ub);
%}

%% First specify the contraints for the weights
init = zeros(N-1,1);
lb = zeros(length(init),1);
ub = ones(length(init),1);
Aeq = ones(1,length(init));
beq = 1;
%{
AM = outcome{1};
%CM = treatment{1};
CM = zeros(T,3,N);
for i = 2:N
    AM = cat(2,AM,outcome{i}); % convert outcomes into an array
    %CM = cat(2,CM,treatment{i});
end

for i=1:N
    CM(:,:,i) = treatment{i}; % convert treatment into an array
end
%}
%% Now let us do a grid search

bgrid = linspace(-0.1,0.1,10);
manugrid = linspace(-0.1,0.1,3);
regrid = linspace(-0.1,0.1,3);
minval = zeros(length(bgrid),length(manugrid),length(regrid),N);
phi = zeros(length(bgrid),length(manugrid),length(regrid),N-1,N);
options = optimoptions('fmincon','Display','off');

disp('Grid search begins.')
parfor i = 1:length(bgrid)
    %disp(['bgrid ',num2str(i)]);
    temp1 = zeros(length(manugrid),length(regrid),N);
    tempphi1 = zeros(length(manugrid),length(regrid),N-1,N);
    for k = 1:length(manugrid)
        %disp(['manugrid ',num2str(k)]);
        tempphi = zeros(length(regrid),N-1,N);
        temp = zeros(length(regrid),N);
        for l = 1:length(regrid)
            %disp(['regrid ',num2str(l)]);
            b = [bgrid(i);manugrid(k);regrid(l)];
            cost_record = zeros(N,1);
            phi_record = zeros(N-1,N);
            for j = 1:N
                y = outcome(:,j);
                d = treatment(:,:,j);
                A = outcome;
                C = treatment;
                A(:,j) = [];
                C(:,:,j) = [];
                fun = @(phi)costfun(y,d,phi,A,C,b);
                phi_record(:,j) = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
                %temp = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
                %phi{j} = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
                cost_record(j,1) = costfun(y,d,phi_record(:,j),A,C,b);
            end
            temp(l,:) = cost_record;
            tempphi(l,:,:) = phi_record;
        end
        temp1(k,:,:) = temp;
        tempphi1(k,:,:,:) = tempphi;
    end
    minval(i,:,:,:) = temp1;
    phi(i,:,:,:,:) = tempphi1;
end
disp('Grid search done.')
summinval = sum(minval,4);
[val,loc] = min(summinval(:));
[idx1,idx2,idx3]=ind2sub(size(summinval),loc);
disp(bgrid(idx1));

%% Repeat the process, with tighter intervals
bgrid = linspace(bgrid(max([idx1-1,1])),bgrid(min([idx1+1,10])),10);
manugrid = linspace(manugrid(max([idx2-1,1])),manugrid(min([idx2+1,3])),3);
regrid = linspace(regrid(max([idx3-1,1])),regrid(min([idx3+1,3])),3);

disp('Grid search begins.')
parfor i = 1:length(bgrid)
    %disp(['bgrid ',num2str(i)]);
    temp1 = zeros(length(manugrid),length(regrid),N);
    tempphi1 = zeros(length(manugrid),length(regrid),N-1,N);
    for k = 1:length(manugrid)
        %disp(['manugrid ',num2str(k)]);
        tempphi = zeros(length(regrid),N-1,N);
        temp = zeros(length(regrid),N);
        for l = 1:length(regrid)
            %disp(['regrid ',num2str(l)]);
            b = [bgrid(i);manugrid(k);regrid(l)];
            cost_record = zeros(N,1);
            phi_record = zeros(N-1,N);
            for j = 1:N
                y = outcome(:,j);
                d = treatment(:,:,j);
                A = outcome;
                C = treatment;
                A(:,j) = [];
                C(:,:,j) = [];
                fun = @(phi)costfun(y,d,phi,A,C,b);
                phi_record(:,j) = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
                %temp = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
                %phi{j} = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
                cost_record(j,1) = costfun(y,d,phi_record(:,j),A,C,b);
            end
            temp(l,:) = cost_record;
            tempphi(l,:,:) = phi_record;
        end
        temp1(k,:,:) = temp;
        tempphi1(k,:,:,:) = tempphi;
    end
    minval(i,:,:,:) = temp1;
    phi(i,:,:,:,:) = tempphi1;
end
disp('Grid search done.')

summinval = sum(minval,4);
[val,loc] = min(summinval(:));
[idx1,idx2,idx3]=ind2sub(size(summinval),loc);
disp(bgrid(idx1));

%{
% Now let's try to write the loop
init = zeros(N-1,1);
lb = zeros(length(init),1);
ub = ones(length(init),1);
Aeq = ones(1,length(init));
beq = 1;
AM = outcome{1};
CM = treatment{1};
%CM = zeros(T,3,N);
for i = 2:N
    AM = cat(2,AM,outcome{i});
    CM = cat(2,CM,treatment{i});
end





bgrid = linspace(-0.1,0.1,30);
phi = zeros(length(bgrid),N-1,N);
minval = zeros(length(bgrid),N);
options = optimoptions('fmincon','Display','off');

parfor i = 1:length(bgrid)
    disp(i)
    temp0 = zeros(N,1);
    temp1 = zeros(N-1,N);
    for j = 1:N
        y = outcome{j};
        d = treatment{j};
        A = AM;
        C = CM;
        A(:,j) = [];
        C(:,j) = [];
        fun = @(phi)costfun1d(y,d,phi,A,C,bgrid(i));
        temp1(:,j) = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
        temp0(j) = costfun1d(y,d,temp1(:,j),A,C,bgrid(i));
    end
    minval(i,:) = temp0;
    phi(i,:,:) = temp1;
end

summinval = sum(minval,2);
[val,loc] = min(summinval(:));
disp(bgrid(loc));

% Now we repeat the process with tighter b grid
bgrid = linspace(bgrid(max([loc-1,1])),bgrid(min([loc+1,10])),30);
phi = zeros(length(bgrid),N-1,N);
minval = zeros(length(bgrid),N);

parfor i = 1:length(bgrid)
    disp(i)
    temp0 = zeros(N,1);
    temp1 = zeros(N-1,N);
    for j = 1:N
        y = outcome{j};
        d = treatment{j};
        A = AM;
        C = CM;
        A(:,j) = [];
        C(:,j) = [];
        fun = @(phi)costfun1d(y,d,phi,A,C,bgrid(i));
        temp1(:,j) = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
        temp0(j) = costfun1d(y,d,temp1(:,j),A,C,bgrid(i));
    end
    minval(i,:) = temp0;
    phi(i,:,:) = temp1;
end

summinval = sum(minval,2);
[val,loc] = min(summinval(:));
disp(bgrid(loc));
%}
%{
% Let us try a parallel version of the code
bgrid = linspace(bgrid(max([idx1-1,1])),bgrid(min([idx1+1,10])),2);
manugrid = linspace(manugrid(max([idx2-1,1])),manugrid(min([idx2+1,10])),2);
regrid = linspace(regrid(max([idx3-1,1])),regrid(min([idx3+1,10])),2);
init = zeros(N-1,1);

for i = 1:length(bgrid)
    disp(['bgrid ',num2str(i)]);
    for k = 1:length(manugrid)
        disp(['manugrid ',num2str(k)]);
        parfor l = 1:length(regrid)
            disp(['regrid ',num2str(l)]);
            b = [bgrid(i);manugrid(k);regrid(l)];
            cost_record = zeros(N,1);
            phi_record = zeros(N-1,N);
            for j = 1:N
                y = outcome{j};
                d = treatment{j};
                A = AM;
                C = CM;
                A(:,j) = [];
                C(:,:,j) = [];
                fun = @(phi)costfun(y,d,phi,A,C,b);
                phi_record(:,j) = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
                %phi{j} = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
                cost_record(j,1) = costfun(y,d,phi_record(:,j),A,C,b);
            end
            minval(i,k,l,:) = cost_record;
        end
    end
end

summinval = sum(minval,4);
[val,loc] = min(summinval(:));
[idx1,idx2,idx3]=ind2sub(size(summinval),loc);
disp(bgrid(idx1));
%}

opt_b = bgrid(idx1); % -0.0086
opt_manu = manugrid(idx2); % 0
opt_re = regrid(idx3); % 0

%% Now let us try to do inference
%{
options = optimoptions('fmincon','Display','off');
phi_null = cell(N,1);
init = zeros(50,1);
for j = 1:N
    y = outcome{j};
    d = treatment{j};
    A = AM;
    C = CM;
    A(:,j) = [];
    C(:,j) = [];
    fun = @(phi)costfun(y,d,phi,A,C,0);
    phi_null{j} = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
end
%}

%% Step 1a: Estimate the constraint coefficients
manugrid = linspace(-0.2,0.2,5);
regrid = linspace(-0.3,0.3,5);
init = zeros(N-1,1);
restrict_phi = zeros(length(manugrid),length(regrid),N-1,N);
restrict_minval = zeros(length(manugrid),length(regrid),N);
options = optimoptions('fmincon','Display','off');


parfor k = 1:length(manugrid)
    %disp(['manugrid ',num2str(k)]);
    tempphi = zeros(length(regrid),N-1,N);
    temp = zeros(length(regrid),N);
    for l = 1:length(regrid)
        %disp(['regrid ',num2str(l)]);
        b = [0;manugrid(k);regrid(l)];
        cost_record = zeros(N,1);
        phi_record = zeros(N-1,N);
        for j = 1:N
            y = outcome(:,j);
            d = treatment(:,:,j);
            A = outcome;
            C = treatment;
            A(:,j) = [];
            C(:,:,j) = [];
            fun = @(phi)costfun(y,d,phi,A,C,b);
            phi_record(:,j) = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
            %temp = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
            %phi{j} = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
            cost_record(j,1) = costfun(y,d,phi_record(:,j),A,C,b);
        end
        temp(l,:) = cost_record;
        tempphi(l,:,:) = phi_record;
    end
    restrict_minval(k,:,:) = temp;
    restrict_phi(k,:,:,:) = tempphi;
end



restrict_summinval = sum(restrict_minval,3);
[restrict_val,restrict_loc] = min(restrict_summinval(:));
[inf1,inf2]=ind2sub(size(restrict_summinval),restrict_loc);
disp([inf1,inf2]);
restrict_b = [0;manugrid(inf1);regrid(inf2)];
restrict_weight = squeeze(restrict_phi(inf1,inf2,:,:));

%% Step 1b: Repeat the process with tighter grid
manugrid = linspace(manugrid(max([inf1-1,1])),manugrid(min([inf1+1,5])),5);
regrid = linspace(regrid(max([inf2-1,1])),regrid(min([inf2+1,5])),5);
%manugrid = linspace(-0.1,-0.1,1);
%regrid = linspace(-0.1875,-0.1875,1);
init = zeros(N-1,1);
restrict_phi = zeros(length(manugrid),length(regrid),N-1,N);
restrict_minval = zeros(length(manugrid),length(regrid),N);
options = optimoptions('fmincon','Display','off');

parfor k = 1:length(manugrid)
    %disp(['manugrid ',num2str(k)]);
    tempphi = zeros(length(regrid),N-1,N);
    temp = zeros(length(regrid),N);
    for l = 1:length(regrid)
        %disp(['regrid ',num2str(l)]);
        b = [0;manugrid(k);regrid(l)];
        cost_record = zeros(N,1);
        phi_record = zeros(N-1,N);
        for j = 1:N
            y = outcome(:,j);
            d = treatment(:,:,j);
            A = outcome;
            C = treatment;
            A(:,j) = [];
            C(:,:,j) = [];
            fun = @(phi)costfun(y,d,phi,A,C,b);
            phi_record(:,j) = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
            %temp = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
            %phi{j} = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
            cost_record(j,1) = costfun(y,d,phi_record(:,j),A,C,b);
        end
        temp(l,:) = cost_record;
        tempphi(l,:,:) = phi_record;
    end
    restrict_minval(k,:,:) = temp;
    restrict_phi(k,:,:,:) = tempphi;
end

restrict_summinval = sum(restrict_minval,3);
[restrict_val,restrict_loc] = min(restrict_summinval(:));
[inf1,inf2]=ind2sub(size(restrict_summinval),restrict_loc);
disp([inf1,inf2]);
restrict_b = [0;manugrid(inf1);regrid(inf2)];
restrict_weight = squeeze(restrict_phi(inf1,inf2,:,:));

%% Step 2: Evaluate gradient matrix h
h = zeros(N,T,3);
for j=1:N
    A = outcome;
    C = treatment;
    A(:,j) = [];
    C(:,:,j) = [];
    weight = restrict_weight(:,j);
    temp = zeros(T,3);
    for i=1:T
        other = squeeze(C(i,:,:));
        temp(i,:)=weight'*other';
    end
    DB = zeros(T,N-1);
    for i=1:N-1
        DB(:,i)=C(:,:,i)*restrict_b;
    end
    h(j,:,:) = ((treatment(:,:,j)-temp).*(outcome(:,j)-treatment(:,:,j)*restrict_b-(A-DB)*weight))/T;
end

g = cell(3,1);
for i=1:3
    g{i} = squeeze(h(:,:,i));
end



%% Step 3: Create Wald Statistics
s = zeros(N,3);
for i=1:3
    for j = 1:N-1
        intercept = ones(T,1);
        temp = g{i};
        y = temp(j,:)';
        x = cat(2,intercept,temp(j+1:N,:)');
        B = x\y;
        yhat = x*B;
        resid = y-yhat;
        s(j,i) = mean(resid);
    end
end

% Create the Wald Statistic
sm = mean(s,1);
cov_matrix = (s-sm)'*(s-sm)/(N-1);
wald = sm*pinv(cov_matrix)*sm';

%% Step 4: Perturb using Rademacher function
D = 999;
perturb = randsrc(N,D);
waldt = zeros(D,1);
for i = 1:D
    sd = perturb(:,i).*s;
    sdm = mean(sd,1);
    cov_matrixd = (sd-sdm)'*(sd-sdm)/(N-1);
    waldt(i) = sdm*pinv(cov_matrix)*sdm';
end

p = sum(waldt > wald)/D; 

%{
h = zeros(N,T);
for j = 1:N
    A = AM;
    C = CM;
    A(:,j) = [];
    C(:,j) = [];
    h(j,:) = (treatment{j}-C*phi_null{j}).*(outcome{j} - A*phi_null{j})/T;
end


s = zeros(N,1);
for j = 1:N-1
    intercept = ones(T,1);
    y = h(j,:)';
    x = cat(2,intercept,h(j+1:N,:)');
    B = x\y;
    yhat = x*B;
    resid = y-yhat;
    s(j,1) = mean(resid);
end
% Create the Wald statistic
sm = mean(s);
cov_matrix = (s-sm)'*(s-sm)/(N-1);
wald = sm^2/cov_matrix;

% We now need to perturb using Rademacher function
D = 999;
perturb = randsrc(N,D);
waldt = zeros(D,1);
cov_matrixd = zeros(D,1);
for i = 1:D
    sd = perturb(:,i).*s;
    sdm = mean(sd);
    cov_matrixd(i) = (sd-sdm)'*(sd-sdm)/(N-1);
    waldt(i) = sdm^2/cov_matrixd(i);
end

p = sum(waldt > wald)/D;
%}


%% Show the fit of the generalized synthetic control
% The idea is that we mimic the synthetic control plot to show the average
% fit
% We need to do it for each states
% The output should be plots of the comparisons
%{
state_list = ["Alabama","Alaska","Arizona","Arkansas","California",...
              "Colorado","Connecticut","Delaware","District of Columbia","Florida",...
              "Georgia","Hawaii","Idaho","Illinois","Indiana",...
              "Iowa","Kansas","Kentucky","Louisiana","Maine",...
              "Maryland","Massachusetts","Michigan","Minnesota","Mississippi",...
              "Missouri","Montana","Nebraska","Nevada","New Hampshire",...
              "New Jersey","New Mexico","New York","North Carolina","North Dakota",...
              "Ohio","Oklahoma","Oregon","Pennsylvania",...
              "Rhode Island","South Carolina","South Dakota","Tennessee","Texas",...
              "Utah","Vermont","Virginia","Washington","West Virginia",...
              "Wisconsin","Wyoming"];
%state_list_alter = ["Alaska","Arizona","Arkansas","Colorado",...
%                    "Connecticut","District of
correlation = output_plot(outcome,treatment,trial_weight,coef,state_list);
correlation_result = calculate_correlation(correlation,N);

% Get counterfactual weight
counterfactual_weight = ones(N-1,N);
for i=1:N
    counterfactual_weight(:,i) = counterfactual_weight(:,i)/50;
end
correlation_counterfactual = output_plot(outcome,treatment,counterfactual_weight,coef_counterfactual,state_list);
correlation_counterfactual_result = calculate_correlation(correlation_counterfactual,N);

% Only include the states that changes minimum wage often



%% Draft Space
coef = [opt_b;opt_manu;opt_re];
coef_counterfactual = [-0.015;-0.051;-0.13];
trial_weight = calculate_weight(outcome,treatment,coef,N,init,Aeq,beq,lb,ub,options);

[trial_y1,trial_y2] = gsc_fit(outcome,treatment,trial_weight,coef,7,N);
trial_y1_t = hpfilter(trial_y1,14400);
trial_y2_t = hpfilter(trial_y2,14400);
f = figure('visible','off');
plot(trial_y1_t)
hold on
plot(trial_y2_t)
ylabel('Mobility Rate');
grid on
grid minor
legend('California','Control');
%legend boxoff;
title('California');
saveas(f,strcat('trial','_fig.png'));
trial_correlation = output_plot(outcome,treatment,trial_weight,coef);
%}
%% 1D Case
%{
outcome = cell(56,1);
treatment = cell(56,1);

for i = 1:56
    for j = 1:length(data)
        if data(j,2) == i
            outcome{i} = cat(1,outcome{i},data(j,3));  % using the true switch here
            %treatment{i} = cat(1,treatment{i},data(j,[5,7:8]));
            treatment{i} = cat(1,treatment{i},data(j,4));
        end
    end
end

outcome = outcome(~cellfun('isempty',outcome)); %delete empty entries
treatment = treatment(~cellfun('isempty',treatment));
N = length(outcome);
T = length(outcome{1});

init = zeros(N-1,1);
lb = zeros(length(init),1);
ub = ones(length(init),1);
Aeq = ones(1,length(init));
beq = 1;
AM = outcome{1};
CM = treatment{1};
%CM = zeros(T,3,N);
for i = 2:N
    AM = cat(2,AM,outcome{i});
    CM = cat(2,CM,treatment{i});
end





bgrid = linspace(-0.1,0.1,30);
phi = zeros(length(bgrid),N-1,N);
minval = zeros(length(bgrid),N);
options = optimoptions('fmincon','Display','off');

parfor i = 1:length(bgrid)
    disp(i)
    temp0 = zeros(N,1);
    temp1 = zeros(N-1,N);
    for j = 1:N
        y = outcome{j};
        d = treatment{j};
        A = AM;
        C = CM;
        A(:,j) = [];
        C(:,j) = [];
        fun = @(phi)costfun1d(y,d,phi,A,C,bgrid(i));
        temp1(:,j) = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
        temp0(j) = costfun1d(y,d,temp1(:,j),A,C,bgrid(i));
    end
    minval(i,:) = temp0;
    phi(i,:,:) = temp1;
end

summinval = sum(minval,2);
[val,loc] = min(summinval(:));
disp(bgrid(loc));

% Now we repeat the process with tighter b grid
bgrid = linspace(bgrid(max([loc-1,1])),bgrid(min([loc+1,30])),30);
phi = zeros(length(bgrid),N-1,N);
minval = zeros(length(bgrid),N);

parfor i = 1:length(bgrid)
    disp(i)
    temp0 = zeros(N,1);
    temp1 = zeros(N-1,N);
    for j = 1:N
        y = outcome{j};
        d = treatment{j};
        A = AM;
        C = CM;
        A(:,j) = [];
        C(:,j) = [];
        fun = @(phi)costfun1d(y,d,phi,A,C,bgrid(i));
        temp1(:,j) = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
        temp0(j) = costfun1d(y,d,temp1(:,j),A,C,bgrid(i));
    end
    minval(i,:) = temp0;
    phi(i,:,:) = temp1;
end

summinval = sum(minval,2);
[val,loc] = min(summinval(:));
disp(bgrid(loc));

% Inference
options = optimoptions('fmincon','Display','off');
phi_null = cell(N,1);
init = zeros(50,1);
for j = 1:N
    y = outcome{j};
    d = treatment{j};
    A = AM;
    C = CM;
    A(:,j) = [];
    C(:,j) = [];
    fun = @(phi)costfun(y,d,phi,A,C,0);
    phi_null{j} = fmincon(fun,init,[],[],Aeq,beq,lb,ub,[],options);
end

h = zeros(N,T);
for j = 1:N
    A = AM;
    C = CM;
    A(:,j) = [];
    C(:,j) = [];
    h(j,:) = (treatment{j}-C*phi_null{j}).*(outcome{j} - A*phi_null{j})/T;
end


s = zeros(N,1);
for j = 1:N-1
    intercept = ones(T,1);
    y = h(j,:)';
    x = cat(2,intercept,h(j+1:N,:)');
    B = x\y;
    yhat = x*B;
    resid = y-yhat;
    s(j,1) = mean(resid);
end
% Create the Wald statistic
sm = mean(s);
cov_matrix = (s-sm)'*(s-sm)/(N-1);
wald = sm^2/cov_matrix;

% We now need to perturb using Rademacher function
D = 999;
perturb = randsrc(N,D);
waldt = zeros(D,1);
cov_matrixd = zeros(D,1);
for i = 1:D
    sd = perturb(:,i).*s;
    sdm = mean(sd);
    cov_matrixd(i) = (sd-sdm)'*(sd-sdm)/(N-1);
    waldt(i) = sdm^2/cov_matrixd(i);
end

p = sum(waldt > wald)/D;    
%} 
