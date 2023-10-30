%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reproducible file accompanying chapter 2 of the 
% Thesis "Iterative algorithms for sparse and low-rank recovery from atypical measurements"
% by Srinivas Subramanian  
% Chapter 2: ITERATIVE ALGORITHMS FOR SPARSE RECOVERY FROM MEASUREMENT MATRICES SATISFYING AN l1 RESTRICTED ISOMETRY PROPERTY 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 1a: Influence of the parameters k and kappa on MHTP1 and MHTP2



clear variables; clc;

% define the problem sizes
N = 1000 ; 
m = 500 ; 
s = 50 ; 

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 6 ;    % number of measurement matrices
n_Vec = 5 ;    % number of sparse vectors per measurement matrix
tol_suc = 1e-5 ;
kappamax = floor(N/s) ; % max value of kappa tested 
kmax = 30 ; % max value of k tested 
Res_MHTP1 = zeros(kmax,kappamax) ; 
Res_MHTP2 = zeros(kmax,kappamax) ; 
Time_MHTP1 = zeros(kmax,kappamax) ; 
Time_MHTP2 = zeros(kmax,kappamax) ; 

for meas = 1:n_Meas
    % tracks experiment progress
    fprintf('Number of measurement schemes tested = %d \n',meas-1) ;
    % define random A 
    A = randlap([m,N],1)/m ;
    for vect = 1:n_Vec
        % the sparse vector to be recovered
        x = zeros(N,1);
        supp = sort(randperm(N,s));       
        x(supp) = randn(s,1); 
        % its measurement vector
        y = A*x;
        for k = 1:kmax
            for kappa = 1:kappamax
                % tracks experiment progress
                fprintf('k=%d, kappa=%d \n', k, kappa);
                % reconstruction by MHTP1
                tic ; 
                x_MHTP1 = MHTP(A,y,s,1,k,kappa) ; 
                t_MHTP1 = toc ; 
                % record computation time 
                Time_MHTP1(k,kappa) = Time_MHTP1(k,kappa) + t_MHTP1 ; 
                % record success if relative error smaller than tolerance
                Res_MHTP1(k,kappa) = Res_MHTP1(k,kappa) + (norm(x-x_MHTP1) < tol_suc*norm(x)) ;
                % reconstruction by MHTP2
                tic ; 
                x_MHTP2 = MHTP(A,y,s,2,k,kappa) ; 
                t_MHTP2 = toc ; 
                % record computation time 
                Time_MHTP2(k,kappa) = Time_MHTP2(k,kappa) + t_MHTP2 ; 
                % record success if relative error smaller than tolerance
                Res_MHTP2(k,kappa) = Res_MHTP2(k,kappa) + (norm(x-x_MHTP2) < tol_suc*norm(x)) ;
            end
        end
    end
end
t1a = sum(sum(Time_MHTP1)) + sum(sum(Time_MHTP2)) ;

save('Exp1a.mat')

%% Visualization of the results of Experiment 1a

try load('Exp1a.mat')
catch
    load('Exp1a_default.mat')
end

figure
surf(Res_MHTP1'/n_Meas/n_Vec,'FaceAlpha',0.5);
xlabel('k','FontSize',22);
ylabel('kappa','FontSize',22);
zlabel('Frequency of success','FontSize',22);
title(strcat('Recovery success for MHTP1 (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
    '\newline','N = ',num2str(N), ', m = ',num2str(m), ', s = ',num2str(s)), 'FontSize',16);

figure
surf(Res_MHTP2'/n_Meas/n_Vec,'FaceAlpha',0.5);
xlabel('k','FontSize',22);
ylabel('kappa','FontSize',22);
zlabel('Frequency of success','FontSize',22);
title(strcat('Recovery success for MHTP2 (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
    '\newline','N = ',num2str(N), ', m = ',num2str(m), ', s = ',num2str(s)), 'FontSize',16);

% Observe the above plots and check the Res arrays to determine
% kappa_cutoff for region of complete success 
kappa_cutoff = 7 ; 
% Check timings on success region to determine the best parameters 

Time_MHTP1_suc = Time_MHTP1(:,1:kappa_cutoff) ; 
figure
surf(Time_MHTP1_suc'/n_Meas/n_Vec,'FaceAlpha',0.5);
xlabel('k','FontSize',22);
ylabel('kappa','FontSize',22);
zlabel('Time (in sec)','FontSize',22);
title(strcat('Execution time for MHTP1 in success region (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
    '\newline','N = ',num2str(N), ', m = ',num2str(m), ', s = ',num2str(s)), 'FontSize',16);

Time_MHTP2_suc = Time_MHTP2(:,1:kappa_cutoff) ; 
figure
surf(Time_MHTP2_suc'/n_Meas/n_Vec,'FaceAlpha',0.5);
xlabel('k','FontSize',22);
ylabel('kappa','FontSize',22);
zlabel('Time (in sec)','FontSize',22);
title(strcat('Execution time for MHTP2 in success region (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
    '\newline','N = ',num2str(N), ', m = ',num2str(m), ', s = ',num2str(s)), 'FontSize',16);

% It is clear that higher the kappa the slower it is, therefore we choose
% kappa = 1 as the default parameter 
% For MHTP1, it seems like a larger value of k is better 
% For MHTP2, it seems like a smaller value of k is better 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 1b: Determine the optimal k for MHTP1 and MHTP2. 
% Fixing kappa = 1, vary k = 1:50, with a larger number of
% trials to get a more accurate picture than the previous experiment 

clear variables; clc;

% define the problem sizes
N = 1000 ; 
m = 500 ; 
s = 50 ; 

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 10 ;    % number of measurement matrices
n_Vec = 10 ;    % number of sparse vectors per measurement matrix
kmax = 50 ; % max value of k tested 
Time_MHTP1 = zeros(kmax,1) ; 
Time_MHTP2 = zeros(kmax,1) ; 

for meas = 1:n_Meas
    % tracks experiment progress
    fprintf('Number of measurement schemes tested = %d \n',meas-1) ;
    % define random A 
    A = randlap([m,N],1)/m ;
    for vect = 1:n_Vec
        % the sparse vector to be recovered
        x = zeros(N,1);
        supp = sort(randperm(N,s));       
        x(supp) = randn(s,1); 
        % its measurement vector
        y = A*x;
        for k = 1:kmax        
                % tracks experiment progress
                fprintf('k = %d \n', k);
                % reconstruction by MHTP1
                tic ; 
                x_MHTP1 = MHTP(A,y,s,1,k) ; 
                t_MHTP1 = toc ; 
                % record computation time 
                Time_MHTP1(k) = Time_MHTP1(k) + t_MHTP1 ; 
                
                % reconstruction by MHTP2
                tic ; 
                x_MHTP2 = MHTP(A,y,s,2,k) ; 
                t_MHTP2 = toc ; 
                % record computation time 
                Time_MHTP2(k) = Time_MHTP2(k) + t_MHTP2 ;   
        end
    end
end
t1b = sum(Time_MHTP1) + sum(Time_MHTP2) ;

save('Exp1b.mat')

%% Visualization of the results of Experiment 1b

try load('Exp1b.mat')
catch
    load('Exp1b_default.mat')
end

figure
plot(Time_MHTP1/n_Meas/n_Vec,'k:x');
xlabel('Parameter k','FontSize',22);
ylabel('Time (in sec)','FontSize',22);
title(strcat('Execution time for MHTP1 with kappa = 1 (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
    '\newline','N = ',num2str(N), ', m = ',num2str(m), ', s = ',num2str(s)), 'FontSize',16);

% larger the k, the faster MHTP1 is. Therefore choose k as large as needed.
% Taking k too large will cause MHTP1 to converge before l1 minimization
% and effectively becomes MIHT. We can choose k = 50, since for this regime 
% MIHT takes about 56 iterations to converge

figure
plot(Time_MHTP2/n_Meas/n_Vec,'r-d');
xlabel('Parameter k','FontSize',22);
ylabel('Time (in sec)','FontSize',22);
title(strcat('Execution time for MHTP2 with kappa = 1 (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
    '\newline','N = ',num2str(N), ', m = ',num2str(m), ', s = ',num2str(s)), 'FontSize',16);

% smallest value of k = 1 is the best for MHTP2 
% MHTP2 seems to get slower as k increases but between k = 23:31, MHTP2
% gets faster for some reason. In any case, k = 1 is fastest by far. 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 1c: Effect of parameter kappa on MIHT 

clear variables; clc;

% define the problem sizes
N = 1000 ; 
m = 500 ; 
s = 50 ; 

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 10 ;    % number of measurement matrices
n_Vec = 10 ;    % number of sparse vectors per measurement matrix
tol_suc = 1e-5 ;
kappamax = floor(N/s) ; % max value of kappa tested 
Time_MIHT = zeros(kappamax,1) ; 
Res_MIHT = zeros(kappamax,1) ; 

for meas = 1:n_Meas
    % tracks experiment progress
    fprintf('Number of measurement schemes tested = %d \n',meas-1) ;
    % define random A 
    A = randlap([m,N],1)/m ;
    for vect = 1:n_Vec
        % the sparse vector to be recovered
        x = zeros(N,1);
        supp = sort(randperm(N,s));       
        x(supp) = randn(s,1); 
        % its measurement vector
        y = A*x;
        for kappa = 1:kappamax        
                % tracks experiment progress
                fprintf('kappa = %d \n', kappa);
                % reconstruction by MIHT
                tic ; 
                x_MIHT = MHTP(A,y,s,0,[],kappa) ; 
                t_MIHT = toc ; 
                % record computation time 
                Time_MIHT(kappa) = Time_MIHT(kappa) + t_MIHT ; 
                % record success if relative error smaller than tolerance
                Res_MIHT(kappa) = Res_MIHT(kappa) + (norm(x-x_MIHT) < tol_suc*norm(x)) ;
                  
        end
    end
end
t1c = sum(Time_MIHT) ;

save('Exp1c.mat')

%% Visualization of the results of Experiment 1c

try load('Exp1c.mat')
catch
    load('Exp1c_default.mat')
end

figure
plot(Res_MIHT/n_Meas/n_Vec,'b--o');
xlabel('Parameter kappa','FontSize',22);
ylabel('Frequency of success','FontSize',22);
title(strcat('Recovery success for MIHT (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
    '\newline','N = ',num2str(N), ', m = ',num2str(m), ', s = ',num2str(s)), 'FontSize',16);

% kappa_cutoff for region of complete success 
kappa_cutoff = 3 ; 
% Check timings on success region to determine the best kappa 
% Only 3 values to check, so we just display it instead of plotting 
fprintf('Execution time for MIHT in the success region kappa = 1:%d \n',kappa_cutoff) ;
display(Time_MIHT(1:kappa_cutoff)'/n_Meas/n_Vec);
% Values are increasing with kappa, the minimum is at kappa = 1 
% Therefore the default value of kappa for MIHT is kappa = 1