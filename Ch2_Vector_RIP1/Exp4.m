%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reproducible file accompanying chapter 2 of the 
% Thesis "Iterative algorithms for sparse and low rank recovery"
% by Srinivas Subramanian  
% Chapter 2: ITERATIVE ALGORITHMS FOR SPARSE RECOVERY FROM MEASUREMENT MATRICES SATISFYING AN l1 RESTRICTED ISOMETRY PROPERTY 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The experiments below require CVX with gurobi 
% This is to solve BP using the function BasisPursuitcvx
% For these dimensions, BaisPursuitcvx is faster than BasispursuitLP (CVX v/s MATLAB's linprog)
% Changing gurobi to mosek does not make much difference 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 4a:  Speed comparison for several algorithms, with Laplacian A, Gaussian x 

clear variables; clc;
cvx_quiet true
cvx_solver gurobi

% define the problem sizes
N = 1500 ; 
m = 500 ; 

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 10 ;    % number of measurement matrices
n_Vec = 10 ;    % number of sparse vectors per measurement matrix
tol_suc = 1e-5 ; 

s_min = 5 ; 
s_inc = 1 ; 
s_max = 50 ; % within the region of success for all shapes  
s_range = s_min:s_inc:s_max ; 
s_num = length(s_range) ; 

Res_MHTP2 = zeros(1,s_num) ; 
Time_MHTP2 = zeros(1,s_num) ; 
Iter_MHTP2 = zeros(1,s_num) ; 

Res_NHTP = zeros(1,s_num) ; 
Time_NHTP = zeros(1,s_num) ; 
Iter_NHTP = zeros(1,s_num) ; 

Res_MIHT = zeros(1,s_num) ; 
Time_MIHT = zeros(1,s_num) ; 
Iter_MIHT = zeros(1,s_num) ; 

Res_BP = zeros(1,s_num) ; 
Time_BP = zeros(1,s_num) ; 

Res_MHTP1 = zeros(1,s_num) ; 
Time_MHTP1 = zeros(1,s_num) ; 
Iter_MHTP1 = zeros(1,s_num) ; 
nIt_MHTP1 = zeros(1,s_num) ; 
k = 50 ; 

for t = 1:s_num
            % tracks experiment progress 
          fprintf('Test with t=%d \n',t) ;
          s = s_range(t) ; 
          for meas = 1:n_Meas
                fprintf('Number of measurement schemes tested = %d at t = %d \n',meas-1,t) ;
                % define random A 
                A = randlap([m,N],1)/m ;
                for vect = 1:n_Vec
                    % the sparse x to be recovered 
                    x = zeros(N,1) ;
                    supp = sort(randperm(N,s)) ;       
                    x(supp) = randn(s,1) ; 
                    norm_x = norm(x) ;
                    y = A*x ; 
                    % perform the reconstructions and record results 
                    tic ; 
                    [x_MHTP2,i_MHTP2] = MHTP(A,y,s,2) ; 
                    t_MHTP2 = toc ;     
                    Time_MHTP2(t) = Time_MHTP2(t) + t_MHTP2 ;   
                    Iter_MHTP2(t) = Iter_MHTP2(t) + i_MHTP2 ; 
                    Res_MHTP2(t) = Res_MHTP2(t) + (norm(x-x_MHTP2) < tol_suc*norm_x) ;
                    
                    tic ; 
                    [x_NHTP,i_NHTP] = NHTP(A,y,s) ; 
                    t_NHTP = toc ;     
                    Time_NHTP(t) = Time_NHTP(t) + t_NHTP ;   
                    Iter_NHTP(t) = Iter_NHTP(t)+ i_NHTP ;  
                    Res_NHTP(t) = Res_NHTP(t) + (norm(x-x_NHTP) < tol_suc*norm_x) ; 
                    
                    tic ; 
                    [x_MIHT,i_MIHT] = MHTP(A,y,s,0) ; 
                    t_MIHT = toc ;     
                    Time_MIHT(t) = Time_MIHT(t) + t_MIHT ;   
                    Iter_MIHT(t) = Iter_MIHT(t)+ i_MIHT ;  
                    Res_MIHT(t) = Res_MIHT(t) + (norm(x-x_MIHT) < tol_suc*norm_x) ; 
                    
                    tic ;  
                    x_BP = BasisPursuitcvx(A,y) ; 
                    t_BP = toc ; 
                    Time_BP(t) = Time_BP(t) + t_BP ; 
                    Res_BP(t) = Res_BP(t) + (norm(x-x_BP) < tol_suc*norm_x) ; 
                         
                    tic ; 
                    [x_MHTP1,i_MHTP1,n_MHTP1] = MHTP(A,y,s,1,k) ; 
                    t_MHTP1 = toc ;     
                    Time_MHTP1(t) = Time_MHTP1(t) + t_MHTP1 ;   
                    Iter_MHTP1(t) = Iter_MHTP1(t) + i_MHTP1 ; 
                    nIt_MHTP1(t) = nIt_MHTP1(t) + n_MHTP1 ;
                    Res_MHTP1(t) = Res_MHTP1(t) + (norm(x-x_MHTP1) < tol_suc*norm_x) ;
                    
                end
          end
end    

t4a = sum(Time_MHTP2)+sum(Time_NHTP)+sum(Time_MIHT)+sum(Time_BP)+sum(Time_MHTP1) ; 

save('Exp4a.mat')

%% Visualization of the results of Experiment 4a

try load('Exp4a.mat')
catch
    load('Exp4a_default.mat')
end

figure
        plot(s_range, Iter_MHTP2/n_Meas/n_Vec,'k:x',...
            s_range, Iter_NHTP/n_Meas/n_Vec,'b--o',...
            s_range, Iter_MIHT/n_Meas/n_Vec,'r-d',...
            s_range, Iter_MHTP1/n_Meas/n_Vec, 'g-.+') ;
        legend('MHTP2','NHTP','MIHT','MHTP1','Location','northwest');
        title(strcat('Number of iterations (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Gaussian'),'FontSize',16);
    xlabel('Sparsity s','FontSize',22);
    ylabel('Number of iterations','FontSize',22); 

figure
        plot(s_range, Iter_MHTP2/n_Meas/n_Vec,'k:x',...
            s_range, nIt_MHTP1/n_Meas/n_Vec, 'g-.+') ;
        legend('MHTP2','MHTP1','Location','northwest') ;
        title(strcat('Number of outer iterations (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Gaussian'),'FontSize',16);
    xlabel('Sparsity s','FontSize',22);
    ylabel('Number of minimizations','FontSize',22); 
        
    
    
figure
        plot(s_range, Time_MHTP2/n_Meas/n_Vec,'k:x',...
            s_range, Time_NHTP/n_Meas/n_Vec,'b--o',...
            s_range, Time_MIHT/n_Meas/n_Vec,'r-d',...
            s_range, Time_BP/n_Meas/n_Vec, 'y--v',...
            s_range, Time_MHTP1/n_Meas/n_Vec, 'g-.+') ;
        legend('MHTP2','NHTP','MIHT','BP','MHTP1','Location','northwest');
        title(strcat('Execution time (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Gaussian'),'FontSize',16);
    xlabel('Sparsity s','FontSize',22);
    ylabel('Time T (in sec)','FontSize',22); 
    
    % plots displaying execution times again, but in logarithmic scale
        logs = log(s_range);
        
        logT_MHTP2 = log(Time_MHTP2/n_Meas/n_Vec);
        coefsT_MHTP2 = polyfit(logs,logT_MHTP2,1);
        logT_MHTP2_fitted = coefsT_MHTP2(1)*logs + coefsT_MHTP2(2);
        
        logT_NHTP = log(Time_NHTP/n_Meas/n_Vec);
        coefsT_NHTP = polyfit(logs,logT_NHTP,1);
        logT_NHTP_fitted = coefsT_NHTP(1)*logs + coefsT_NHTP(2);
        
        logT_MIHT = log(Time_MIHT/n_Meas/n_Vec);
        coefsT_MIHT = polyfit(logs,logT_MIHT,1);
        logT_MIHT_fitted = coefsT_MIHT(1)*logs + coefsT_MIHT(2);
        
        logT_BP = log(Time_BP/n_Meas/n_Vec);
        coefsT_BP = polyfit(logs,logT_BP,1);
        logT_BP_fitted = coefsT_BP(1)*logs + coefsT_BP(2);
        
        logT_MHTP1 = log(Time_MHTP1/n_Meas/n_Vec);
        coefsT_MHTP1 = polyfit(logs,logT_MHTP1,1);
        logT_MHTP1_fitted = coefsT_MHTP1(1)*logs + coefsT_MHTP1(2);
        
        figure
        plot(logs,logT_MHTP2,'kx',logs,logT_MHTP2_fitted,'k:',...
            logs,logT_NHTP,'bo',logs,logT_NHTP_fitted,'b--',...
            logs,logT_MIHT,'rd',logs,logT_MIHT_fitted,'r-',...
            logs,logT_BP,'yv',logs,logT_BP_fitted,'y--v',...
            logs,logT_MHTP1,'g+',logs,logT_MHTP1_fitted,'g-.+');
        legend('MHTP2', strcat('slope=', num2str(coefsT_MHTP2(1),'%4.2f')),...
            'NHTP', strcat('slope=', num2str(coefsT_NHTP(1),'%4.2f')),...
            'MIHT', strcat('slope=', num2str(coefsT_MIHT(1),'%4.2f')),...
            'BP', strcat('slope=', num2str(coefsT_BP(1),'%4.2f')),...
            'MHTP1', strcat('slope=', num2str(coefsT_MHTP1(1),'%4.2f')),...
            'Location', 'northwest');
        title(strcat('Execution time (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Gaussian'),'FontSize',16);
        xlabel('Logarithm of s','FontSize',22);
        ylabel('Logarithm of T','FontSize',22);
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 4b:  Performance comparison for several algorithms, with Laplacian A, Rademacher x 


clear variables; clc;
cvx_quiet true
cvx_solver gurobi

% define the problem sizes
N = 1500 ; 
m = 500 ; 

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 10 ;    % number of measurement matrices
n_Vec = 10 ;    % number of sparse vectors per measurement matrix
tol_suc = 1e-5 ; 

s_min = 5 ; 
s_inc = 1 ; 
s_max = 50 ; % within the region of success for all shapes  
s_range = s_min:s_inc:s_max ; 
s_num = length(s_range) ; 

Res_MHTP2 = zeros(1,s_num) ; 
Time_MHTP2 = zeros(1,s_num) ; 
Iter_MHTP2 = zeros(1,s_num) ; 

Res_NHTP = zeros(1,s_num) ; 
Time_NHTP = zeros(1,s_num) ; 
Iter_NHTP = zeros(1,s_num) ; 

Res_MIHT = zeros(1,s_num) ; 
Time_MIHT = zeros(1,s_num) ; 
Iter_MIHT = zeros(1,s_num) ; 

Res_BP = zeros(1,s_num) ; 
Time_BP = zeros(1,s_num) ; 

Res_MHTP1 = zeros(1,s_num) ; 
Time_MHTP1 = zeros(1,s_num) ; 
Iter_MHTP1 = zeros(1,s_num) ; 
nIt_MHTP1 = zeros(1,s_num) ; 
k = 50 ; 

for t = 1:s_num
            % tracks experiment progress 
          fprintf('Test with t=%d \n',t) ;
          s = s_range(t) ; 
          for meas = 1:n_Meas
                fprintf('Number of measurement schemes tested = %d at t = %d \n',meas-1,t) ;
                % define random A 
                A = randlap([m,N],1)/m ;
                for vect = 1:n_Vec
                    % the sparse x to be recovered 
                    x = zeros(N,1) ;
                    supp = sort(randperm(N,s)) ;       
                    x(supp) = sign(randn(s,1)) ; % x = sgn(randn)
                    norm_x = norm(x) ;
                    y = A*x ; 
                    % perform the reconstructions and record results 
                    tic ; 
                    [x_MHTP2,i_MHTP2] = MHTP(A,y,s,2) ; 
                    t_MHTP2 = toc ;     
                    Time_MHTP2(t) = Time_MHTP2(t) + t_MHTP2 ;   
                    Iter_MHTP2(t) = Iter_MHTP2(t) + i_MHTP2 ; 
                    Res_MHTP2(t) = Res_MHTP2(t) + (norm(x-x_MHTP2) < tol_suc*norm_x) ;
                    
                    tic ; 
                    [x_NHTP,i_NHTP] = NHTP(A,y,s) ; 
                    t_NHTP = toc ;     
                    Time_NHTP(t) = Time_NHTP(t) + t_NHTP ;   
                    Iter_NHTP(t) = Iter_NHTP(t)+ i_NHTP ;  
                    Res_NHTP(t) = Res_NHTP(t) + (norm(x-x_NHTP) < tol_suc*norm_x) ; 
                    
                    tic ; 
                    [x_MIHT,i_MIHT] = MHTP(A,y,s,0) ; 
                    t_MIHT = toc ;     
                    Time_MIHT(t) = Time_MIHT(t) + t_MIHT ;   
                    Iter_MIHT(t) = Iter_MIHT(t)+ i_MIHT ;  
                    Res_MIHT(t) = Res_MIHT(t) + (norm(x-x_MIHT) < tol_suc*norm_x) ; 
                    
                     tic ;  
                    x_BP = BasisPursuitcvx(A,y) ; 
                    t_BP = toc ; 
                    Time_BP(t) = Time_BP(t) + t_BP ; 
                    Res_BP(t) = Res_BP(t) + (norm(x-x_BP) < tol_suc*norm_x) ; 
                              
                    tic ; 
                    [x_MHTP1,i_MHTP1,n_MHTP1] = MHTP(A,y,s,1,k) ; 
                    t_MHTP1 = toc ;     
                    Time_MHTP1(t) = Time_MHTP1(t) + t_MHTP1 ;   
                    Iter_MHTP1(t) = Iter_MHTP1(t) + i_MHTP1 ; 
                    nIt_MHTP1(t) = nIt_MHTP1(t) + n_MHTP1 ;
                    Res_MHTP1(t) = Res_MHTP1(t) + (norm(x-x_MHTP1) < tol_suc*norm_x) ;
                    
                end
          end
end    

t4b = sum(Time_MHTP2)+sum(Time_NHTP)+sum(Time_MIHT)+sum(Time_BP)+sum(Time_MHTP1) ; 

save('Exp4b.mat')


%% Visualization of the results of Experiment 4b

try load('Exp4b.mat')
catch
    load('Exp4b_default.mat')
end

figure
       plot(s_range, Iter_MHTP2/n_Meas/n_Vec,'k:x',...
            s_range, Iter_NHTP/n_Meas/n_Vec,'b--o',...
            s_range, Iter_MIHT/n_Meas/n_Vec,'r-d',...
            s_range, Iter_MHTP1/n_Meas/n_Vec, 'g-.+') ;
        legend('MHTP2','NHTP','MIHT','MHTP1','Location','northwest');
        title(strcat('Number of iterations (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Flat'),'FontSize',16);
    xlabel('Sparsity s','FontSize',22);
    ylabel('Number of iterations','FontSize',22); 
    
    
figure
        plot(s_range, Iter_MHTP2/n_Meas/n_Vec,'k:x',...
            s_range, nIt_MHTP1/n_Meas/n_Vec, 'g-.+') ;
        legend('MHTP2','MHTP1','Location','northwest') ;
        title(strcat('Number of outer iterations (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Flat'),'FontSize',16);
    xlabel('Sparsity s','FontSize',22);
    ylabel('Number of minimizations','FontSize',22); 
    
figure
           plot(s_range, Time_MHTP2/n_Meas/n_Vec,'k:x',...
            s_range, Time_NHTP/n_Meas/n_Vec,'b--o',...
            s_range, Time_MIHT/n_Meas/n_Vec,'r-d',...
            s_range, Time_BP/n_Meas/n_Vec, 'y--v',...
            s_range, Time_MHTP1/n_Meas/n_Vec, 'g-.+') ;
        legend('MHTP2','NHTP','MIHT','BP','MHTP1','Location','northwest');
        title(strcat('Execution time (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Flat'),'FontSize',16);
    xlabel('Sparsity s','FontSize',22);
    ylabel('Time T (in sec)','FontSize',22); 
    
    % plots displaying execution times again, but in logarithmic scale
        logs = log(s_range);
        
        logT_MHTP2 = log(Time_MHTP2/n_Meas/n_Vec);
        coefsT_MHTP2 = polyfit(logs,logT_MHTP2,1);
        logT_MHTP2_fitted = coefsT_MHTP2(1)*logs + coefsT_MHTP2(2);
        
        logT_NHTP = log(Time_NHTP/n_Meas/n_Vec);
        coefsT_NHTP = polyfit(logs,logT_NHTP,1);
        logT_NHTP_fitted = coefsT_NHTP(1)*logs + coefsT_NHTP(2);
        
        logT_MIHT = log(Time_MIHT/n_Meas/n_Vec);
        coefsT_MIHT = polyfit(logs,logT_MIHT,1);
        logT_MIHT_fitted = coefsT_MIHT(1)*logs + coefsT_MIHT(2);
        
          logT_BP = log(Time_BP/n_Meas/n_Vec);
        coefsT_BP = polyfit(logs,logT_BP,1);
        logT_BP_fitted = coefsT_BP(1)*logs + coefsT_BP(2);
        
        logT_MHTP1 = log(Time_MHTP1/n_Meas/n_Vec);
        coefsT_MHTP1 = polyfit(logs,logT_MHTP1,1);
        logT_MHTP1_fitted = coefsT_MHTP1(1)*logs + coefsT_MHTP1(2);
        
        figure
           plot(logs,logT_MHTP2,'kx',logs,logT_MHTP2_fitted,'k:',...
            logs,logT_NHTP,'bo',logs,logT_NHTP_fitted,'b--',...
            logs,logT_MIHT,'rd',logs,logT_MIHT_fitted,'r-',...
            logs,logT_BP,'yv',logs,logT_BP_fitted,'y--v',...
            logs,logT_MHTP1,'g+',logs,logT_MHTP1_fitted,'g-.+');
        legend('MHTP2', strcat('slope=', num2str(coefsT_MHTP2(1),'%4.2f')),...
            'NHTP', strcat('slope=', num2str(coefsT_NHTP(1),'%4.2f')),...
            'MIHT', strcat('slope=', num2str(coefsT_MIHT(1),'%4.2f')),...
            'BP', strcat('slope=', num2str(coefsT_BP(1),'%4.2f')),...
            'MHTP1', strcat('slope=', num2str(coefsT_MHTP1(1),'%4.2f')),...
            'Location', 'northwest');
        title(strcat('Execution time (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Flat'),'FontSize',16);
        xlabel('Logarithm of s','FontSize',22);
        ylabel('Logarithm of T','FontSize',22);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 4c:  Performance comparison for several algorithms, with Laplacian A, linear decreasing entries x 


clear variables; clc;
cvx_quiet true
cvx_solver gurobi

% define the problem sizes
N = 1500 ; 
m = 500 ; 

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 10 ;    % number of measurement matrices
n_Vec = 10 ;    % number of sparse vectors per measurement matrix
tol_suc = 1e-5 ; 

s_min = 5 ; 
s_inc = 1 ; 
s_max = 50 ; % within the region of success for all shapes  
s_range = s_min:s_inc:s_max ; 
s_num = length(s_range) ; 

Res_MHTP2 = zeros(1,s_num) ; 
Time_MHTP2 = zeros(1,s_num) ; 
Iter_MHTP2 = zeros(1,s_num) ; 

Res_NHTP = zeros(1,s_num) ; 
Time_NHTP = zeros(1,s_num) ; 
Iter_NHTP = zeros(1,s_num) ; 

Res_MIHT = zeros(1,s_num) ; 
Time_MIHT = zeros(1,s_num) ; 
Iter_MIHT = zeros(1,s_num) ; 

Res_BP = zeros(1,s_num) ; 
Time_BP = zeros(1,s_num) ; 

Res_MHTP1 = zeros(1,s_num) ; 
Time_MHTP1 = zeros(1,s_num) ; 
Iter_MHTP1 = zeros(1,s_num) ; 
nIt_MHTP1 = zeros(1,s_num) ; 
k = 50 ; 

for t = 1:s_num
            % tracks experiment progress 
          fprintf('Test with t=%d \n',t) ;
          s = s_range(t) ; 
          for meas = 1:n_Meas
                fprintf('Number of measurement schemes tested = %d at t = %d \n',meas-1,t) ;
                % define random A 
                A = randlap([m,N],1)/m ;
                for vect = 1:n_Vec
                    % the sparse x to be recovered 
                    x = zeros(N,1) ;
                    supp = randperm(N,s) ;       
                    x(supp) = 1+(1-[1:s])/s ; 
                    norm_x = norm(x) ;
                    y = A*x ; 
                    % perform the reconstructions and record results 
                    tic ; 
                    [x_MHTP2,i_MHTP2] = MHTP(A,y,s,2) ; 
                    t_MHTP2 = toc ;     
                    Time_MHTP2(t) = Time_MHTP2(t) + t_MHTP2 ;   
                    Iter_MHTP2(t) = Iter_MHTP2(t) + i_MHTP2 ; 
                    Res_MHTP2(t) = Res_MHTP2(t) + (norm(x-x_MHTP2) < tol_suc*norm_x) ;
                    
                    tic ; 
                    [x_NHTP,i_NHTP] = NHTP(A,y,s) ; 
                    t_NHTP = toc ;     
                    Time_NHTP(t) = Time_NHTP(t) + t_NHTP ;   
                    Iter_NHTP(t) = Iter_NHTP(t)+ i_NHTP ;  
                    Res_NHTP(t) = Res_NHTP(t) + (norm(x-x_NHTP) < tol_suc*norm_x) ; 
                    
                    tic ; 
                    [x_MIHT,i_MIHT] = MHTP(A,y,s,0) ; 
                    t_MIHT = toc ;     
                    Time_MIHT(t) = Time_MIHT(t) + t_MIHT ;   
                    Iter_MIHT(t) = Iter_MIHT(t)+ i_MIHT ;  
                    Res_MIHT(t) = Res_MIHT(t) + (norm(x-x_MIHT) < tol_suc*norm_x) ; 
                    
                     tic ;  
                    x_BP = BasisPursuitcvx(A,y) ; 
                    t_BP = toc ; 
                    Time_BP(t) = Time_BP(t) + t_BP ; 
                    Res_BP(t) = Res_BP(t) + (norm(x-x_BP) < tol_suc*norm_x) ; 
                         
                         
                    tic ; 
                    [x_MHTP1,i_MHTP1,n_MHTP1] = MHTP(A,y,s,1,k) ; 
                    t_MHTP1 = toc ;     
                    Time_MHTP1(t) = Time_MHTP1(t) + t_MHTP1 ;   
                    Iter_MHTP1(t) = Iter_MHTP1(t) + i_MHTP1 ; 
                    nIt_MHTP1(t) = nIt_MHTP1(t) + n_MHTP1 ;
                    Res_MHTP1(t) = Res_MHTP1(t) + (norm(x-x_MHTP1) < tol_suc*norm_x) ;
                    
                end
          end
end    


t4c = sum(Time_MHTP2)+sum(Time_NHTP)+sum(Time_MIHT)+sum(Time_BP)+sum(Time_MHTP1) ; 

save('Exp4c.mat')


%% Visualization of the results of Experiment 4c
try load('Exp4c.mat')
catch
    load('Exp4c_default.mat')
end

figure
       plot(s_range, Iter_MHTP2/n_Meas/n_Vec,'k:x',...
            s_range, Iter_NHTP/n_Meas/n_Vec,'b--o',...
            s_range, Iter_MIHT/n_Meas/n_Vec,'r-d',...
            s_range, Iter_MHTP1/n_Meas/n_Vec, 'g-.+') ;
        legend('MHTP2','NHTP','MIHT','MHTP1','Location','northwest');
        title(strcat('Number of iterations (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Linear'),'FontSize',16);
    xlabel('Sparsity s','FontSize',22);
    ylabel('Number of iterations','FontSize',22); 

        
figure
        plot(s_range, Iter_MHTP2/n_Meas/n_Vec,'k:x',...
            s_range, nIt_MHTP1/n_Meas/n_Vec, 'g-.+') ;
        legend('MHTP2','MHTP1','Location','northwest') ;
        title(strcat('Number of outer iterations (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Linear'),'FontSize',16);
    xlabel('Sparsity s','FontSize',22);
    ylabel('Number of minimizations','FontSize',22); 
    
figure
            plot(s_range, Time_MHTP2/n_Meas/n_Vec,'k:x',...
            s_range, Time_NHTP/n_Meas/n_Vec,'b--o',...
            s_range, Time_MIHT/n_Meas/n_Vec,'r-d',...
            s_range, Time_BP/n_Meas/n_Vec, 'y--v',...
            s_range, Time_MHTP1/n_Meas/n_Vec, 'g-.+') ;
        legend('MHTP2','NHTP','MIHT','BP','MHTP1','Location','northwest');
        title(strcat('Execution time (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Linear'),'FontSize',16);
    xlabel('Sparsity s','FontSize',22);
    ylabel('Time T (in sec)','FontSize',22); 
    
% plots displaying execution times again, but in logarithmic scale
        logs = log(s_range);
        
        logT_MHTP2 = log(Time_MHTP2/n_Meas/n_Vec);
        coefsT_MHTP2 = polyfit(logs,logT_MHTP2,1);
        logT_MHTP2_fitted = coefsT_MHTP2(1)*logs + coefsT_MHTP2(2);
        
        logT_NHTP = log(Time_NHTP/n_Meas/n_Vec);
        coefsT_NHTP = polyfit(logs,logT_NHTP,1);
        logT_NHTP_fitted = coefsT_NHTP(1)*logs + coefsT_NHTP(2);
        
        logT_MIHT = log(Time_MIHT/n_Meas/n_Vec);
        coefsT_MIHT = polyfit(logs,logT_MIHT,1);
        logT_MIHT_fitted = coefsT_MIHT(1)*logs + coefsT_MIHT(2);
        
          logT_BP = log(Time_BP/n_Meas/n_Vec);
        coefsT_BP = polyfit(logs,logT_BP,1);
        logT_BP_fitted = coefsT_BP(1)*logs + coefsT_BP(2);
        
        logT_MHTP1 = log(Time_MHTP1/n_Meas/n_Vec);
        coefsT_MHTP1 = polyfit(logs,logT_MHTP1,1);
        logT_MHTP1_fitted = coefsT_MHTP1(1)*logs + coefsT_MHTP1(2);
        
        figure
          plot(logs,logT_MHTP2,'kx',logs,logT_MHTP2_fitted,'k:',...
            logs,logT_NHTP,'bo',logs,logT_NHTP_fitted,'b--',...
            logs,logT_MIHT,'rd',logs,logT_MIHT_fitted,'r-',...
            logs,logT_BP,'yv',logs,logT_BP_fitted,'y--v',...
            logs,logT_MHTP1,'g+',logs,logT_MHTP1_fitted,'g-.+');
        legend('MHTP2', strcat('slope=', num2str(coefsT_MHTP2(1),'%4.2f')),...
            'NHTP', strcat('slope=', num2str(coefsT_NHTP(1),'%4.2f')),...
            'MIHT', strcat('slope=', num2str(coefsT_MIHT(1),'%4.2f')),...
            'BP', strcat('slope=', num2str(coefsT_BP(1),'%4.2f')),...
            'MHTP1', strcat('slope=', num2str(coefsT_MHTP1(1),'%4.2f')),...
            'Location', 'northwest');
        title(strcat('Execution time (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Linear'),'FontSize',16);
        xlabel('Logarithm of s','FontSize',22);
        ylabel('Logarithm of T','FontSize',22);    