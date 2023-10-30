%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reproducible file accompanying Chapter 3 of the 
% Thesis "Iterative algorithms for sparse and low-rank recovery from atypical measurements"
% by Srinivas Subramanian
% Chapter 3: ITERATIVE ALGORITHMS FOR LOW-RANK RECOVERY FROM MEASUREMENT MAPS SATISFYING AN l1 RANK RESTRICTED ISOMETRY PROPERTY
% 
% Written by Srinivas Subramanian and Simon Foucart 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVX [2] is needed to perform nuclear norm minimizations


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 1a: Influence of the parameters s and t on MIHT 

clear all; clc;

% define the problem sizes
N1 = 20;
N2 = 20;
r = 3;
m = 5*r*max(N1,N2);

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 10;    % number of rank-one projections
n_Matr = 10;    % number of low-rank matrices per rank-one projection
tol_suc = 1e-3;
f = floor(min(N1,N2)/r);
Res = zeros(f,f);

tic;
for meas = 1:n_Meas
    % tracks experiment progress
    fprintf('Number of measurement schemes tested = %d \n',meas-1);
    % define rank-one projections via A=[a_1|...|a_m] and B=[b_1|...|b_m]
    A = randn(N1,m)/sqrt(m);
    B = randn(N2,m)/sqrt(m);
    for matr = 1:n_Matr
        % the low-rank matrix to be recovered
        X = randn(N1,r)*randn(r,N2);
        % its measurement vector
        y = sum(A.*(X*B))';
        for s_over_r = 1:f
            for t_over_r = 1:f
                % tracks experiment progress
                fprintf('s/r=%d, t/r=%d \n', s_over_r, t_over_r);
                % reconstruction by modified IHT
                X_MIHT = MIHT(A,B,y,r,s_over_r,t_over_r);
                % record success if relative error smaller than tolerance
                Res(s_over_r,t_over_r) = Res(s_over_r,t_over_r) + ...
                    ( norm(X-X_MIHT,'fro') < tol_suc*norm(X,'fro') );
            end
        end
    end
end
t1 = toc;

save('Exp1a.mat')


%% Visualization of the results of Experiment 1a

try load('Exp1a.mat')
catch
    load('Exp1a_default.mat')
end

figure
surf(Res'/n_Meas/n_Matr,'FaceAlpha',0.5);
xlabel('s/r ','FontSize',22);
ylabel('t/r','FontSize',22);
zlabel('Frequency of success','FontSize',22);
title(strcat('Recovery success (averaged over', 32, num2str(n_Meas*n_Matr),' trials)',...
    '\newline','N_1=',num2str(N1), ', N_2=',num2str(N2), ', r=',num2str(r), ', m=',num2str(m)),'FontSize',16);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Experiment 1b: Influence of the parameters s and t on MHTP

clear variables; clc;

% define the problem sizes
N1 = 20;
N2 = 20;
r = 3;
m = 5*r*max(N1,N2);

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 10;    % number of rank-one projections
n_Matr = 10;    % number of low-rank matrices per rank-one projection
tol_suc = 1e-3;
f = floor(min(N1,N2)/r);
Res = zeros(f,f);

tic;
for meas = 1:n_Meas
    % tracks experiment progress
    fprintf('Number of measurement schemes tested = %d \n',meas-1);
    % define rank-one projections via A=[a_1|...|a_m] and B=[b_1|...|b_m]
    A = randn(N1,m)/sqrt(m);
    B = randn(N2,m)/sqrt(m);
    for matr = 1:n_Matr
        % the low-rank matrix to be recovered
        X = randn(N1,r)*randn(r,N2);
        % its measurement vector
        y = sum(A.*(X*B))';
        for s_over_r = 1:f
            for t_over_r = 1:f
                % tracks experiment progress
                fprintf('s/r=%d, t/r=%d \n', s_over_r, t_over_r);
                % reconstruction by HTP
                X_HTP = MIHT(A,B,y,r,s_over_r,t_over_r,1);
                % record success if relative error smaller than tolerance
                Res(s_over_r,t_over_r) = Res(s_over_r,t_over_r) + ...
                    ( norm(X-X_HTP,'fro') < tol_suc*norm(X,'fro') );
            end
        end
    end
end
t1 = toc;

save('Exp1b.mat')


%% Visualization of the results of Experiment 1b

try load('Exp1b.mat')
catch
    load('Exp1b_default.mat')
end

figure
surf(Res'/n_Meas/n_Matr,'FaceAlpha',0.5);
xlabel('s/r ','FontSize',22);
ylabel('t/r','FontSize',22);
zlabel('Frequency of success','FontSize',22);
title(strcat('Recovery success (averaged over', 32, num2str(n_Meas*n_Matr),' trials)',...
    '\newline','N_1=',num2str(N1), ', N_2=',num2str(N2), ', r=',num2str(r), ', m=',num2str(m)),'FontSize',16);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 2: Performance comparison for several algorithms:
% NNM, MIHT, MHTP and NHTP

clear variables; clc;
cvx_quiet true
cvx_solver mosek

% define the problem sizes
dim_min = 30;                          % minimal dimension to be tested 
dim_max = 40;                          % maximal dimension to be tested
dim_inc = 10;                          % increment for the dimension
n_dim = floor((dim_max-dim_min)/dim_inc) + 1; % number of dimensions tested
rk_min = 2;                            % minimal rank to be tested
rk_max = 3;                            % maximal rank to be tested
rk_inc = 1;                            % increment for the rank
n_rk = floor((rk_max-rk_min)/rk_inc) + 1;     % number of ranks tested

% set numbers of random trials, success tolerance
n_Meas = 10;        % number of rank-one projections
n_Matr = 5;        % number of low-rank matrices per rank-one projection
tol_suc = 1e-3;

tic;
for i = 1:n_dim
    N = dim_min + (i-1)*dim_inc;
    N1 = N; N2 = N;
    for j = 1:n_rk
        r = rk_min + (j-1)*rk_inc;
        % define numbers of measurements
        m_inc = 4;
        m_min = 2*r*N;
        m_max = 7*r*N;
        n_m = floor((m_max-m_min)/m_inc) + 1;
        m_range{i,j} = m_min:m_inc:m_max;
        Res_MIHT_def{i,j} = zeros(1,n_m);
        Res_HTP{i,j} = zeros(1,n_m);
        Res_NIHT{i,j} = zeros(1,n_m);
        Res_NNM{i,j} = zeros(1,n_m);
        for k = 1:n_m
            m = m_min + (k-1)*m_inc;
            % tracks experiment progress 
            fprintf('Test with N=%d, r=%d, and m=%d (start: m=%d, end: m=%d) \n',...
                N, r, m, m_min, m_max);
            for meas = 1:n_Meas
                % define rank-one projections via A=[a_1|...|a_m] and B=[b_1|...|b_m]
                A = randn(N1,m)/sqrt(m);
                B = randn(N2,m)/sqrt(m);
                for matr = 1:n_Matr
                    % the low-rank matrix to be recovered
                    X = randn(N1,r)*randn(r,N2);
                    norm_X = norm(X,'fro');
                    % its measurement vector
                    y = sum(A.*(X*B))';
                    % perform the reconstructions
                    X_MIHT_def = MIHT(A,B,y,r);
                    X_HTP = MIHT(A,B,y,r,[],[],1);
                    X_NIHT = NIHT(A,B,y,r);
                    X_NNM = NNM(A,B,y);
                    % success if relative error smaller than tolerance
                    Res_MIHT_def{i,j}(k) = Res_MIHT_def{i,j}(k) + ...
                        ( norm(X-X_MIHT_def,'fro') < tol_suc*norm_X );
                    Res_HTP{i,j}(k) = Res_HTP{i,j}(k) + ...
                        ( norm(X-X_HTP,'fro') < tol_suc*norm_X );
                    Res_NIHT{i,j}(k) = Res_NIHT{i,j}(k) + ...
                        ( norm(X-X_NIHT,'fro') < tol_suc*norm_X );
                    Res_NNM{i,j}(k) = Res_NNM{i,j}(k) + ...
                        ( norm(X-X_NNM,'fro') < tol_suc*norm_X );
                end
            end
        end    
    end
end
t2 = toc;

save('Exp2.mat')


%%  Visualization of the results of Experiment 2

try load('Exp2.mat')
catch
    load('Exp2_default.mat')
end

% plot figures for each test
for i = 1:n_dim
    N = dim_min + (i-1)*dim_inc;
    for j = 1:n_rk
        r = rk_min + (j-1)*rk_inc;
        figure
        plot(m_range{i,j},Res_NNM{i,j}/n_Meas/n_Matr,'k:x',...
            m_range{i,j},Res_HTP{i,j}/n_Meas/n_Matr,'b--o',...
            m_range{i,j},Res_MIHT_def{i,j}/n_Meas/n_Matr,'r-d',...
            m_range{i,j},Res_NIHT{i,j}/n_Meas/n_Matr,'g-.+');
        legend('NNM','MHTP','MIHT (default)','NIHT','Location','southeast');
        title(strcat('Recovery success (averaged over', 32, num2str(n_Meas*n_Matr),' trials)',...
            '\newline','N_1=',num2str(N), ', N_2=',num2str(N), ...
            ', r=',num2str(r)),'FontSize',16);
    xlabel('Number of measurements (m)','FontSize',22);
    ylabel('Frequency of success','FontSize',22); 
    end
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 3: Speed comparison for several algorithms:
% NNM, MIHT, MHTP and NHTP

clear variables; clc;
cvx_quiet true
cvx_solver mosek

% define the problem sizes
dim_min = [22 40];                  % minimal dimensions to be tested
dim_max = [50 70];                  % maximal dimensions to be tested
dim_inc = [1  1];                   % increments in the dimension 
rk_min = 2;                         % minimal rank to be teste
rk_max = 3;                         % maximal rank to be tested
rk_inc = 1;                         % increment in the rank
n_rk = floor((rk_max-rk_min)/rk_inc) + 1;  % number of ranks tested
 
%
n_Meas = 10;                         % number of rank-one projections
n_Matr = 5;                         % number of low-rank matrices per rank-one projection
tol_suc = 1e-3;

for i = 1:n_rk
    r = rk_min + (i-1)*rk_inc;
    for j = 1:size(dim_min,2)
        N_range{i,j} = dim_min(j):dim_inc(j):dim_max(j);
        L = length(N_range{i,j});
        % Tables to hold results for each {i,j}
        Time_MIHT_def{i,j} = zeros(1,L);
        Time_HTP{i,j} = zeros(1,L);
        Time_NIHT{i,j} = zeros(1,L);
        Time_NNM{i,j} = zeros(1,L);
        Iter_MIHT_def{i,j} = zeros(1,L);
        Iter_HTP{i,j} = zeros(1,L);
        Iter_NIHT{i,j} = zeros(1,L);
        for k = 1:L
            N = N_range{i,j}(k);
            % tracks experiment progress
            fprintf('Test with r=%d and N=%d (start: N=%d, end: N=%d) \n',...
                r, N, dim_min(j), dim_max(j));
            N1 = N; N2 = N;
            c = 7;
            m = c*r*N;   % to enforce a regime of successful recovery
            for meas = 1:n_Meas
                % define rank-one projections via A=[a_1|...|a_m] and B=[b_1|...|b_m]
                A = randn(N1,m)/sqrt(m);
                B = randn(N2,m)/sqrt(m);
                for matr = 1:n_Matr
                    % the low-rank matrix to be recovered
                    X = randn(N1,r)*randn(r,N2);
                    norm_X = norm(X,'fro');
                    % its measurement vector
                    y = sum(A.*(X*B))';
                    % perform the reconstructions 
                    % add times and iterations only if successful
                    tic; 
                    [X_MIHT_def,n_MIHT_def] = MIHT(A,B,y,r); 
                    t_MIHT_def = toc;
                    suc = ( norm(X-X_MIHT_def,'fro') < tol_suc*norm_X );
                    Time_MIHT_def{i,j}(k) = Time_MIHT_def{i,j}(k) + t_MIHT_def*suc;
                    Iter_MIHT_def{i,j}(k) = Iter_MIHT_def{i,j}(k) + n_MIHT_def*suc;
                    tic;
                    [X_HTP,n_HTP] = MIHT(A,B,y,r,[],[],1);
                    t_HTP = toc;
                    suc = ( norm(X-X_HTP,'fro') < tol_suc*norm_X );
                    Time_HTP{i,j}(k) = Time_HTP{i,j}(k) + t_HTP*suc;
                    Iter_HTP{i,j}(k) = Iter_HTP{i,j}(k) + n_HTP*suc;
                    tic;
                    [X_NIHT,n_NIHT] = NIHT(A,B,y,r);
                    t_NIHT = toc;
                    suc = ( norm(X-X_NIHT,'fro') < tol_suc*norm_X );
                    Time_NIHT{i,j}(k) = Time_NIHT{i,j}(k) + t_NIHT*suc;
                    Iter_NIHT{i,j}(k) = Iter_NIHT{i,j}(k) + n_NIHT*suc;
                    tic;
                    X_NNM = NNM(A,B,y);
                    t_NNM = toc;
                    suc = ( norm(X-X_NNM,'fro') < tol_suc*norm_X );
                    Time_NNM{i,j}(k) = Time_NNM{i,j}(k) + t_NNM*suc;
                end
            end
        end
    end
end
% total reconstruction time (does not include creation of matrices, etc)
t3 = sum(sum(cellfun(@sum,Time_MIHT_def)))+...
    sum(sum(cellfun(@sum,Time_HTP)))+...
    sum(sum(cellfun(@sum,Time_NIHT)))+...
    sum(sum(cellfun(@sum,Time_NNM)));

save('Exp3.mat')


%%  Visualization of the results of Experiment 3

try load('Exp3.mat')
catch
    load('Exp3_default.mat')
end

for i = 1:n_rk
    r = rk_min + (i-1)*rk_inc;
    for j = 1:size(dim_min,2)
        % plots displaying numbers of iterations
        figure
        plot(N_range{i,j},Iter_HTP{i,j}/n_Meas/n_Matr,'b--o',...
            N_range{i,j},Iter_MIHT_def{i,j}/n_Meas/n_Matr,'r-d',...
            N_range{i,j},Iter_NIHT{i,j}/n_Meas/n_Matr,'g-.+');
        legend('MHTP','MIHT (default)','NIHT','Location', 'northwest');
        title(strcat(...
            'Number of iterations (averaged over ', 32, num2str(n_Meas*n_Matr),' trials)',...
            '\newline', 'r=', num2str(r), ', m =', num2str(c), 'rN'),'FontSize',16);
        xlabel('Dimension N','FontSize',22);
        ylabel('Number of iterations','FontSize',22);  
        % plots displaying execution times
        figure
        plot(N_range{i,j},Time_NNM{i,j}/n_Meas/n_Matr,'k:x',...
            N_range{i,j},Time_HTP{i,j}/n_Meas/n_Matr,'b--o',...
            N_range{i,j},Time_MIHT_def{i,j}/n_Meas/n_Matr,'r-d',...
            N_range{i,j},Time_NIHT{i,j}/n_Meas/n_Matr,'g-.+');
        legend('NNM','MHTP','MIHT (default)','NIHT');
        title(strcat(...
            'Execution time (averaged over ', 32, num2str(n_Meas*n_Matr),' trials)',...
            '\newline', 'r=', num2str(r), ', m =', num2str(c), 'rN'),'FontSize',16);
        xlabel('Dimension N','FontSize',22);
        ylabel('Time T (in sec)','FontSize',22);
        % plots displaying execution times again, but in logarithmic scale
        logN = log(N_range{i,j});
        logT_NNM = log(Time_NNM{i,j}/n_Meas/n_Matr);
        coefsT_NNM = polyfit(logN,logT_NNM,1);
        logT_NNM_fitted = coefsT_NNM(1)*logN+ coefsT_NNM(2);
        logT_HTP = log(Time_HTP{i,j}/n_Meas/n_Matr);
        coefsT_HTP = polyfit(logN,logT_HTP,1);
        logT_HTP_fitted = coefsT_HTP(1)*logN+ coefsT_HTP(2);
        logT_MIHT_def = log(Time_MIHT_def{i,j}/n_Meas/n_Matr);
        coefsT_MIHT_def = polyfit(logN,logT_MIHT_def,1);
        logT_MIHT_def_fitted = coefsT_MIHT_def(1)*logN+ coefsT_MIHT_def(2);
        logT_NIHT = log(Time_NIHT{i,j}/n_Meas/n_Matr);
        coefsT_NIHT = polyfit(logN,logT_NIHT,1);
        logT_NIHT_fitted = coefsT_NIHT(1)*logN+ coefsT_NIHT(2);
        figure
        plot(logN,logT_NNM,'kx',logN,logT_NNM_fitted,'k:',...
            logN,logT_HTP,'bo',logN,logT_HTP_fitted,'b--',...
            logN,logT_MIHT_def,'rd',logN,logT_MIHT_def_fitted,'r-',...
            logN,logT_NIHT,'g+',logN,logT_NIHT_fitted,'g-.');
        legend('NNM', strcat('slope=', num2str(coefsT_NNM(1),'%4.2f')),...
            'MHTP', strcat('slope=', num2str(coefsT_HTP(1),'%4.2f')),...
            'MIHT (default)', strcat('slope=', num2str(coefsT_MIHT_def(1),'%4.2f')),...
            'NIHT', strcat('slope=', num2str(coefsT_NIHT(1),'%4.2f')),...
            'Location', 'northwest');
        title(strcat(...
            'Execution time (averaged over ', 32, num2str(n_Meas*n_Matr),' trials)',...
            '\newline', 'r=', num2str(r), ', m =', num2str(c), 'rN'),'FontSize',16);
        xlabel('Logarithm of N','FontSize',22);
        ylabel('Logarithm of T','FontSize',22);
    end
end
   

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 4: Robustness
% Reconstruction error as a function of the L1-norm of the measurement error

clear variables; clc;

% define the problem sizes
N1 = 40;
N2 = 40;
r = 3;
m = 7*r*max(N1,N2);

% prepare the loop
n_Meas = 10;
n_Matr = 5;
norme_max = 1e-1;
k_max = 20;
Err_HTP = zeros(k_max,1);
Err_MIHT_def = zeros(k_max,1);
Err_NIHT = zeros(k_max,1);

tic ;
for meas = 1:n_Meas
    % tracks experiment progress
    fprintf('Number of measurement schemes tested = %d \n',meas-1);
    A = randn(N1,m)/sqrt(m);
    B = randn(N2,m)/sqrt(m);
    for matr = 1:n_Matr
        X = randn(N1,r)*randn(r,N2);
        norm_X = norm(X,'fro');
        for k=1:k_max
            norme = (k/k_max)*norme_max;
            e = randn(m,1);
            e = norme/norm(e,1)*e;
            y = sum(A.*(X*B))' + e;
            X_HTP = MIHT(A,B,y,r,[],[],1); 
            e_HTP = norm(X-X_HTP,'fro')/norm_X;
            Err_HTP(k) = Err_HTP(k)+e_HTP;
            X_MIHT_def = MIHT(A,B,y,r);
            e_MIHT_def = norm(X-X_MIHT_def,'fro')/norm_X;
            Err_MIHT_def(k) = Err_MIHT_def(k)+e_MIHT_def;
            X_NIHT = NIHT(A,B,y,r); 
            e_NIHT = norm(X-X_NIHT,'fro')/norm_X;
            Err_NIHT(k) = Err_NIHT(k)+e_NIHT;
        end
    end 
end
Err_HTP = Err_HTP/n_Meas/n_Matr;
Err_MIHT_def = Err_MIHT_def/n_Meas/n_Matr;
Err_NIHT = Err_NIHT/n_Meas/n_Matr;

t4 = toc;
save('Exp4.mat');


%% Visualization of the results of Experiment 4

try load('Exp4.mat')
catch
    load('Exp4_default.mat')
end

figure
plot((1:k)/k_max*norme_max,Err_HTP,'b--o',...
    (1:k)/k_max*norme_max,Err_MIHT_def,'r-d',...
    (1:k)/k_max*norme_max,Err_NIHT,'g-.+');
legend('MHTP','MIHT (default)','NIHT','Location','NorthWest') ;
title(strcat('Recovery error (averaged over', 32,num2str(n_Meas*n_Matr),' trials)',...
    '\newline','N_1=',num2str(N1), ', N_2=',num2str(N2),', r=',num2str(r),', m=',num2str(m)),'FontSize',16);
xlabel('$\|e\|_1$','Interpreter','Latex','FontSize',22);
ylabel('$\| X-X_n \|_F$','Interpreter','Latex','FontSize',22);


%% This experiment was not included in the thesis
% Experiment 5a: Number of iterations versus rank for a fixed number of
% measurements ( m = c*r_max*N ) for MHTP & NIHT
% Here we chose N = 80, m = 3.5*20*80, r varies from 1 to 20. 

clear variables; clc;
cvx_quiet true
cvx_solver mosek

% define the problem sizes
r_min = [1];                  % minimal rank to be tested
r_max = [20];                  % maximal rank  to be tested
r_inc = [1  1];                   % increments in the rank 
N_min = 80;                         % minimal dimension to be tested
N_max = 80;                         % maximal dimension to be tested
N_inc = 20;                         % increment in the dimension
n_N = floor((N_max-N_min)/N_inc) + 1;  % number of dimensions tested 
%
n_Meas = 5;                         % number of rank-one projections
n_Matr = 5;                         % number of low-rank matrices per rank-one projection
tol_suc = 1e-3;

for i = 1:n_N
    N = N_min + (i-1)*N_inc;
    for j = 1:size(r_min,2)
        r_range{i,j} = r_min(j):r_inc(j):r_max(j);
        L = length(r_range{i,j});
        % Tables to hold results for each {i,j}
        Time_HTP{i,j} = zeros(1,L);
        Time_NIHT{i,j} = zeros(1,L);
        Iter_HTP{i,j} = zeros(1,L);
        Iter_NIHT{i,j} = zeros(1,L);
        for k = 1:L
            r = r_range{i,j}(k);
            % tracks experiment progress
            fprintf('Test with N=%d and r=%d (start: r=%d, end: r=%d) \n',...
                N, r, r_min(j), r_max(j));
            N1 = N; N2 = N;
            c = 3.5;
            m = c*r_max(j)*N;   % to enforce a regime of successful recovery
            for meas = 1:n_Meas
                % define rank-one projections via A=[a_1|...|a_m] and B=[b_1|...|b_m]
                A = randn(N1,m)/sqrt(m);
                B = randn(N2,m)/sqrt(m);
                for matr = 1:n_Matr
                    % the low-rank matrix to be recovered
                    X = randn(N1,r)*randn(r,N2);
                    norm_X = norm(X,'fro');
                    % its measurement vector
                    y = sum(A.*(X*B))';
                    % perform the reconstructions 
                    % add times and iterations only if successful  
                    tic;
                    [X_HTP,n_HTP] = MIHT(A,B,y,r,[],[],1);
                    t_HTP = toc;
                    suc = ( norm(X-X_HTP,'fro') < tol_suc*norm_X );
                    Time_HTP{i,j}(k) = Time_HTP{i,j}(k) + t_HTP*suc;
                    Iter_HTP{i,j}(k) = Iter_HTP{i,j}(k) + n_HTP*suc;
                    tic;
                    [X_NIHT,n_NIHT] = NIHT(A,B,y,r);
                    t_NIHT = toc;
                    suc = ( norm(X-X_NIHT,'fro') < tol_suc*norm_X );
                    Time_NIHT{i,j}(k) = Time_NIHT{i,j}(k) + t_NIHT*suc;
                    Iter_NIHT{i,j}(k) = Iter_NIHT{i,j}(k) + n_NIHT*suc;
                   
                end
            end
        end
    end
end
% total reconstruction time (does not include creation of matrices, etc)
t5a = sum(sum(cellfun(@sum,Time_HTP)))+...
     sum(sum(cellfun(@sum,Time_NIHT)));
     
save('Exp5a.mat')

%%  Visualization of the results of Experiment 5a

try load('Exp5a.mat')
catch
    load('Exp5a_default.mat')
end

for i = 1:n_N
    N = N_min + (i-1)*N_inc;
    for j = 1:size(r_min,2)
        % plots displaying numbers of iterations
        figure
        coefsIter_HTP = polyfit(r_range{i,j},Iter_HTP{i,j}/n_Meas/n_Matr,1);
        Iter_HTP_fitted = coefsIter_HTP(1)*r_range{i,j} + coefsIter_HTP(2);
        coefsIter_NIHT = polyfit(r_range{i,j},Iter_NIHT{i,j}/n_Meas/n_Matr,1);
        Iter_NIHT_fitted = coefsIter_NIHT(1)*r_range{i,j} + coefsIter_NIHT(2);
         plot(r_range{i,j},Iter_HTP{i,j}/n_Meas/n_Matr,'bo',r_range{i,j},Iter_HTP_fitted,'b--',...
             r_range{i,j},Iter_NIHT{i,j}/n_Meas/n_Matr,'g+',r_range{i,j},Iter_NIHT_fitted,'g-.');
         legend('MHTP', strcat('slope=', num2str(coefsIter_HTP(1),'%4.2f')),...
            'NIHT', strcat('slope=', num2str(coefsIter_NIHT(1),'%4.2f')),...
            'Location', 'northwest');
        title(strcat(...
            'Number of iterations (averaged over ', 32, num2str(n_Meas*n_Matr),' trials)',...
            '\newline', 'N=', num2str(N), ', m =', num2str(c), 'r_{max}N'),'FontSize',16);
        xlabel('Rank r','FontSize',22);
        ylabel('Number of iterations','FontSize',22);  
        % plots displaying execution times
        figure
        plot(r_range{i,j},Time_HTP{i,j}/n_Meas/n_Matr,'b--o',...
            r_range{i,j},Time_NIHT{i,j}/n_Meas/n_Matr,'g-.+');
        legend('MHTP','NIHT');
        title(strcat(...
            'Execution time (averaged over ', 32, num2str(n_Meas*n_Matr),' trials)',...
            '\newline', 'N=', num2str(N), ', m =', num2str(c), 'r_{max}N'),'FontSize',16);
        xlabel('Rank r','FontSize',22);
        ylabel('Time T (in sec)','FontSize',22);
        % plots displaying execution times again, but in logarithmic scale
        logr = log(r_range{i,j});
        logT_HTP = log(Time_HTP{i,j}/n_Meas/n_Matr);
        coefsT_HTP = polyfit(logr,logT_HTP,1);
        logT_HTP_fitted = coefsT_HTP(1)*logr+ coefsT_HTP(2);
        logT_NIHT = log(Time_NIHT{i,j}/n_Meas/n_Matr);
        coefsT_NIHT = polyfit(logr,logT_NIHT,1);
        logT_NIHT_fitted = coefsT_NIHT(1)*logr+ coefsT_NIHT(2);
        figure
        plot(logr,logT_HTP,'bo',logr,logT_HTP_fitted,'b--',...
             logr,logT_NIHT,'g+',logr,logT_NIHT_fitted,'g-.');
        legend('MHTP', strcat('slope=', num2str(coefsT_HTP(1),'%4.2f')),...
            'NIHT', strcat('slope=', num2str(coefsT_NIHT(1),'%4.2f')),...
            'Location', 'northwest');
        title(strcat(...
            'Execution time (averaged over ', 32, num2str(n_Meas*n_Matr),' trials)',...
            '\newline', 'N=', num2str(N), ', m =', num2str(c), 'r_{max}N'),'FontSize',16);
        xlabel('Logarithm of r','FontSize',22);
        ylabel('Logarithm of T','FontSize',22);
    end
end
%% This experiment was not included in the thesis
% Experiment 5b: Number of iterations versus rank for number of 
% measurements varying with rank ( m = c*r*N ) for MHTP & NIHT 
% Below we chose N = 80, m = 3.5*r*80, r varies from 2 to 20. 

clear variables; clc;
cvx_quiet true
cvx_solver mosek

% define the problem sizes
r_min = [2];                  % minimal rank to be tested
r_max = [20];                  % maximal rank to be tested
r_inc = [1  1];                   % increments in the dimension 
N_min = 80;                         % minimal dimension to be tested
N_max = 80;                         % maximal dimension to be tested
N_inc = 20;                         % increment in the dimension
n_N = floor((N_max-N_min)/N_inc) + 1;  % number of dimensions tested 
%
n_Meas = 5;                         % number of rank-one projections
n_Matr = 5;                         % number of low-rank matrices per rank-one projection
tol_suc = 1e-3;

for i = 1:n_N
    N = N_min + (i-1)*N_inc;
    for j = 1:size(r_min,2)
        r_range{i,j} = r_min(j):r_inc(j):r_max(j);
        L = length(r_range{i,j});
        % Tables to hold results for each {i,j}
        Time_HTP{i,j} = zeros(1,L);
        Time_NIHT{i,j} = zeros(1,L);
        Iter_HTP{i,j} = zeros(1,L);
        Iter_NIHT{i,j} = zeros(1,L);
        for k = 1:L
            r = r_range{i,j}(k);
            % tracks experiment progress
            fprintf('Test with N=%d and r=%d (start: r=%d, end: r=%d) \n',...
                N, r, r_min(j), r_max(j));
            N1 = N; N2 = N;
            c = 3.5;
            m = c*r*N;   % to enforce a regime of successful recovery
            for meas = 1:n_Meas
                % define rank-one projections via A=[a_1|...|a_m] and B=[b_1|...|b_m]
                A = randn(N1,m)/sqrt(m);
                B = randn(N2,m)/sqrt(m);
                for matr = 1:n_Matr
                    % the low-rank matrix to be recovered
                    X = randn(N1,r)*randn(r,N2);
                    norm_X = norm(X,'fro');
                    % its measurement vector
                    y = sum(A.*(X*B))';
                    % perform the reconstructions 
                    % add times and iterations only if successful  
                    tic;
                    [X_HTP,n_HTP] = MIHT(A,B,y,r,[],[],1);
                    t_HTP = toc;
                    suc = ( norm(X-X_HTP,'fro') < tol_suc*norm_X );
                    Time_HTP{i,j}(k) = Time_HTP{i,j}(k) + t_HTP*suc;
                    Iter_HTP{i,j}(k) = Iter_HTP{i,j}(k) + n_HTP*suc;
                    tic;
                    [X_NIHT,n_NIHT] = NIHT(A,B,y,r);
                    t_NIHT = toc;
                    suc = ( norm(X-X_NIHT,'fro') < tol_suc*norm_X );
                    Time_NIHT{i,j}(k) = Time_NIHT{i,j}(k) + t_NIHT*suc;
                    Iter_NIHT{i,j}(k) = Iter_NIHT{i,j}(k) + n_NIHT*suc;
                   
                end
            end
        end
    end
end
% total reconstruction time (does not include creation of matrices, etc)
t5b = sum(sum(cellfun(@sum,Time_HTP)))+...
     sum(sum(cellfun(@sum,Time_NIHT)));
     
save('Exp5b.mat')




%%  Visualization of the results of Experiment 5b

try load('Exp5b.mat')
catch
    load('Exp5b_default.mat')
end

for i = 1:n_N
    N = N_min + (i-1)*N_inc;
    for j = 1:size(r_min,2)
        % plots displaying numbers of iterations
        figure
        coefsIter_HTP = polyfit(r_range{i,j},Iter_HTP{i,j}/n_Meas/n_Matr,1);
        Iter_HTP_fitted = coefsIter_HTP(1)*r_range{i,j} + coefsIter_HTP(2);
        coefsIter_NIHT = polyfit(r_range{i,j},Iter_NIHT{i,j}/n_Meas/n_Matr,1);
        Iter_NIHT_fitted = coefsIter_NIHT(1)*r_range{i,j} + coefsIter_NIHT(2);
         plot(r_range{i,j},Iter_HTP{i,j}/n_Meas/n_Matr,'bo',r_range{i,j},Iter_HTP_fitted,'b--',...
             r_range{i,j},Iter_NIHT{i,j}/n_Meas/n_Matr,'g+',r_range{i,j},Iter_NIHT_fitted,'g-.');
         legend('MHTP', strcat('slope=', num2str(coefsIter_HTP(1),'%4.2f')),...
            'NIHT', strcat('slope=', num2str(coefsIter_NIHT(1),'%4.2f')),...
            'Location', 'northwest');
        title(strcat(...
            'Number of iterations (averaged over ', 32, num2str(n_Meas*n_Matr),' trials)',...
            '\newline', 'N=', num2str(N), ', m =', num2str(c), 'rN'),'FontSize',16);
        xlabel('Rank r','FontSize',22);
        ylabel('Number of iterations','FontSize',22);  
        % plots displaying execution times
        figure
        plot(r_range{i,j},Time_HTP{i,j}/n_Meas/n_Matr,'b--o',...
            r_range{i,j},Time_NIHT{i,j}/n_Meas/n_Matr,'g-.+');
        legend('MHTP','NIHT');
        title(strcat(...
            'Execution time (averaged over ', 32, num2str(n_Meas*n_Matr),' trials)',...
            '\newline', 'N=', num2str(N), ', m =', num2str(c), 'rN'),'FontSize',16);
        xlabel('Rank r','FontSize',22);
        ylabel('Time T (in sec)','FontSize',22);
        % plots displaying execution times again, but in logarithmic scale
        logr = log(r_range{i,j});
        logT_HTP = log(Time_HTP{i,j}/n_Meas/n_Matr);
        coefsT_HTP = polyfit(logr,logT_HTP,1);
        logT_HTP_fitted = coefsT_HTP(1)*logr+ coefsT_HTP(2);
        logT_NIHT = log(Time_NIHT{i,j}/n_Meas/n_Matr);
        coefsT_NIHT = polyfit(logr,logT_NIHT,1);
        logT_NIHT_fitted = coefsT_NIHT(1)*logr+ coefsT_NIHT(2);
        figure
        plot(logr,logT_HTP,'bo',logr,logT_HTP_fitted,'b--',...
             logr,logT_NIHT,'g+',logr,logT_NIHT_fitted,'g-.');
        legend('MHTP', strcat('slope=', num2str(coefsT_HTP(1),'%4.2f')),...
            'NIHT', strcat('slope=', num2str(coefsT_NIHT(1),'%4.2f')),...
            'Location', 'northwest');
        title(strcat(...
            'Execution time (averaged over ', 32, num2str(n_Meas*n_Matr),' trials)',...
            '\newline', 'N=', num2str(N), ', m =', num2str(c), 'rN'),'FontSize',16);
        xlabel('Logarithm of r','FontSize',22);
        ylabel('Logarithm of T','FontSize',22);
    end
end


%% References
%
% 1. Thesis of Srinivas Subramanian,
% "Iterative algorithms for sparse and low-rank recovery from atypical measurements"
%
% 2. CVX Research, Inc.,
% "CVX: MATLAB software for disciplined convex programming, version 2.1",
% http://cvxr.com/cvx, 2014.


