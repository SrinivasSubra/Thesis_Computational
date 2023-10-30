%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reproducible file accompanying chapter 2 of the 
% Thesis "Iterative algorithms for sparse and low rank recovery"
% by Srinivas Subramanian
% Chapter 2: ITERATIVE ALGORITHMS FOR SPARSE RECOVERY FROM MEASUREMENT MATRICES SATISFYING AN l1 RESTRICTED ISOMETRY PROPERTY 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 2a:  Performance comparison for several algorithms, with Laplacian A, Gaussian x 


clear variables; clc;

% define the problem sizes
N = 1000 ; 
m = 200 ; 

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 20 ;    % number of measurement matrices
n_Vec = 10 ;    % number of sparse vectors per measurement matrix
tol_suc = 1e-5 ; 
s_max = 80 ; % max value of s 

Res_MHTP2 = zeros(1,s_max) ; 
Time_MHTP2 = zeros(1,s_max) ; 
Iter_MHTP2 = zeros(1,s_max) ; 

Res_NHTP = zeros(1,s_max) ; 
Time_NHTP = zeros(1,s_max) ; 
Iter_NHTP = zeros(1,s_max) ; 

Res_MIHT = zeros(1,s_max) ; 
Time_MIHT = zeros(1,s_max) ; 
Iter_MIHT = zeros(1,s_max) ; 

Res_BP = zeros(1,s_max) ; 
Time_BP = zeros(1,s_max) ; 

Res_MHTP1 = zeros(1,s_max) ; 
Time_MHTP1 = zeros(1,s_max) ; 
Iter_MHTP1 = zeros(1,s_max) ; 
nIt_MHTP1 = zeros(1,s_max) ; 
k = 50 ; 

for s = 1:s_max
            % tracks experiment progress 
          fprintf('Test with s=%d \n',s) ;
          for meas = 1:n_Meas
                fprintf('Number of measurement schemes tested = %d at s = %d \n',meas-1,s) ;
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
                    Time_MHTP2(s) = Time_MHTP2(s) + t_MHTP2 ;   
                    Iter_MHTP2(s) = Iter_MHTP2(s) + i_MHTP2 ; 
                    Res_MHTP2(s) = Res_MHTP2(s) + (norm(x-x_MHTP2) < tol_suc*norm_x) ;
                    
                    tic ; 
                    [x_NHTP,i_NHTP] = NHTP(A,y,s) ; 
                    t_NHTP = toc ;     
                    Time_NHTP(s) = Time_NHTP(s) + t_NHTP ;   
                    Iter_NHTP(s) = Iter_NHTP(s)+ i_NHTP ;  
                    Res_NHTP(s) = Res_NHTP(s) + (norm(x-x_NHTP) < tol_suc*norm_x) ; 
                    
                    tic ; 
                    [x_MIHT,i_MIHT] = MHTP(A,y,s,0) ; 
                    t_MIHT = toc ;     
                    Time_MIHT(s) = Time_MIHT(s) + t_MIHT ;   
                    Iter_MIHT(s) = Iter_MIHT(s)+ i_MIHT ;  
                    Res_MIHT(s) = Res_MIHT(s) + (norm(x-x_MIHT) < tol_suc*norm_x) ; 
                    
                    tic ;  
                    x_BP = BasisPursuitLP(A,y) ; 
                    t_BP = toc ; 
                    Time_BP(s) = Time_BP(s) + t_BP ; 
                    Res_BP(s) = Res_BP(s) + (norm(x-x_BP) < tol_suc*norm_x) ; 
                    
                    tic ; 
                    [x_MHTP1,i_MHTP1,n_MHTP1] = MHTP(A,y,s,1,k) ; 
                    t_MHTP1 = toc ;     
                    Time_MHTP1(s) = Time_MHTP1(s) + t_MHTP1 ;   
                    Iter_MHTP1(s) = Iter_MHTP1(s) + i_MHTP1 ; 
                    nIt_MHTP1(s) = nIt_MHTP1(s) + n_MHTP1 ;
                    Res_MHTP1(s) = Res_MHTP1(s) + (norm(x-x_MHTP1) < tol_suc*norm_x) ;
                    
                end
          end
end    

t2a = sum(Time_MHTP2)+sum(Time_NHTP)+sum(Time_MIHT)+sum(Time_BP)+sum(Time_MHTP1) ; 

save('Exp2a.mat')

%% Visualization of the results of Experiment 2a

try load('Exp2a.mat')
catch
    load('Exp2a_default.mat')
end

S_range = [1:s_max] ; 
figure
        plot(S_range, Res_MHTP2/n_Meas/n_Vec,'k:x',...
            S_range, Res_NHTP/n_Meas/n_Vec,'b--o',...
            S_range, Res_MIHT/n_Meas/n_Vec,'r-d',...
            S_range, Res_BP/n_Meas/n_Vec,'g-.+',...
            S_range, Res_MHTP1/n_Meas/n_Vec, 'y--v') ;
        legend('MHTP2','NHTP','MIHT','BP','MHTP1','Location','northeast');
        title(strcat('Recovery success (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Gaussian '),'FontSize',16);
    xlabel('Sparsity (s)','FontSize',22);
    ylabel('Frequency of success','FontSize',22); 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 2b:  Performance comparison for several algorithms, with Laplacian A, Rademacher x 


clear variables; clc;

% define the problem sizes
N = 1000 ; 
m = 200 ; 

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 20 ;    % number of measurement matrices
n_Vec = 10 ;    % number of sparse vectors per measurement matrix
tol_suc = 1e-5 ; 
s_max = 80 ; % max value of s 

Res_MHTP2 = zeros(1,s_max) ; 
Time_MHTP2 = zeros(1,s_max) ; 
Iter_MHTP2 = zeros(1,s_max) ; 

Res_NHTP = zeros(1,s_max) ; 
Time_NHTP = zeros(1,s_max) ; 
Iter_NHTP = zeros(1,s_max) ; 

Res_MIHT = zeros(1,s_max) ; 
Time_MIHT = zeros(1,s_max) ; 
Iter_MIHT = zeros(1,s_max) ; 

Res_BP = zeros(1,s_max) ; 
Time_BP = zeros(1,s_max) ; 

Res_MHTP1 = zeros(1,s_max) ; 
Time_MHTP1 = zeros(1,s_max) ; 
Iter_MHTP1 = zeros(1,s_max) ; 
nIt_MHTP1 = zeros(1,s_max) ; 
k = 50 ; 

for s = 1:s_max
            % tracks experiment progress 
          fprintf('Test with s=%d \n',s) ;
          for meas = 1:n_Meas
                fprintf('Number of measurement schemes tested = %d at s = %d \n',meas-1,s) ;
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
                    Time_MHTP2(s) = Time_MHTP2(s) + t_MHTP2 ;   
                    Iter_MHTP2(s) = Iter_MHTP2(s) + i_MHTP2 ; 
                    Res_MHTP2(s) = Res_MHTP2(s) + (norm(x-x_MHTP2) < tol_suc*norm_x) ;
                    
                    tic ; 
                    [x_NHTP,i_NHTP] = NHTP(A,y,s) ; 
                    t_NHTP = toc ;     
                    Time_NHTP(s) = Time_NHTP(s) + t_NHTP ;   
                    Iter_NHTP(s) = Iter_NHTP(s)+ i_NHTP ;  
                    Res_NHTP(s) = Res_NHTP(s) + (norm(x-x_NHTP) < tol_suc*norm_x) ; 
                    
                    tic ; 
                    [x_MIHT,i_MIHT] = MHTP(A,y,s,0) ; 
                    t_MIHT = toc ;     
                    Time_MIHT(s) = Time_MIHT(s) + t_MIHT ;   
                    Iter_MIHT(s) = Iter_MIHT(s)+ i_MIHT ;  
                    Res_MIHT(s) = Res_MIHT(s) + (norm(x-x_MIHT) < tol_suc*norm_x) ; 
                    
                    tic ;  
                    x_BP = BasisPursuitLP(A,y) ; 
                    t_BP = toc ; 
                    Time_BP(s) = Time_BP(s) + t_BP ; 
                    Res_BP(s) = Res_BP(s) + (norm(x-x_BP) < tol_suc*norm_x) ; 
                    
                    tic ; 
                    [x_MHTP1,i_MHTP1,n_MHTP1] = MHTP(A,y,s,1) ; 
                    t_MHTP1 = toc ;     
                    Time_MHTP1(s) = Time_MHTP1(s) + t_MHTP1 ;   
                    Iter_MHTP1(s) = Iter_MHTP1(s) + i_MHTP1 ; 
                    nIt_MHTP1(s) = nIt_MHTP1(s) + n_MHTP1 ;
                    Res_MHTP1(s) = Res_MHTP1(s) + (norm(x-x_MHTP1) < tol_suc*norm_x) ;
                    
                end
          end
end    

t2b = sum(Time_MHTP2)+sum(Time_NHTP)+sum(Time_MIHT)+sum(Time_BP)+sum(Time_MHTP1) ; 

save('Exp2b.mat')


%% Visualization of the results of Experiment 2b

try load('Exp2b.mat')
catch
    load('Exp2b_default.mat')
end

S_range = [1:s_max] ; 
figure
        plot(S_range, Res_MHTP2/n_Meas/n_Vec,'k:x',...
            S_range, Res_NHTP/n_Meas/n_Vec,'b--o',...
            S_range, Res_MIHT/n_Meas/n_Vec,'r-d',...
            S_range, Res_BP/n_Meas/n_Vec,'g-.+',...
            S_range, Res_MHTP1/n_Meas/n_Vec, 'y--v') ;
        legend('MHTP2','NHTP','MIHT','BP','MHTP1','Location','northeast');
        title(strcat('Recovery success (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Flat '),'FontSize',16);
    xlabel('Sparsity (s)','FontSize',22);
    ylabel('Frequency of success','FontSize',22); 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 2c:  Performance comparison for several algorithms, with Laplacian A, linear decreasing entries x 


clear variables; clc;

% define the problem sizes
N = 1000 ; 
m = 200 ; 

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 20 ;    % number of measurement matrices
n_Vec = 10 ;    % number of sparse vectors per measurement matrix
tol_suc = 1e-5 ; 
s_max = 80 ; % max value of s 

Res_MHTP2 = zeros(1,s_max) ; 
Time_MHTP2 = zeros(1,s_max) ; 
Iter_MHTP2 = zeros(1,s_max) ; 

Res_NHTP = zeros(1,s_max) ; 
Time_NHTP = zeros(1,s_max) ; 
Iter_NHTP = zeros(1,s_max) ; 

Res_MIHT = zeros(1,s_max) ; 
Time_MIHT = zeros(1,s_max) ; 
Iter_MIHT = zeros(1,s_max) ; 

Res_BP = zeros(1,s_max) ; 
Time_BP = zeros(1,s_max) ; 

Res_MHTP1 = zeros(1,s_max) ; 
Time_MHTP1 = zeros(1,s_max) ; 
Iter_MHTP1 = zeros(1,s_max) ; 
nIt_MHTP1 = zeros(1,s_max) ; 
k = 50 ; 

for s = 1:s_max
            % tracks experiment progress 
          fprintf('Test with s=%d \n',s) ;
          for meas = 1:n_Meas
                fprintf('Number of measurement schemes tested = %d at s = %d \n',meas-1,s) ;
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
                    Time_MHTP2(s) = Time_MHTP2(s) + t_MHTP2 ;   
                    Iter_MHTP2(s) = Iter_MHTP2(s) + i_MHTP2 ; 
                    Res_MHTP2(s) = Res_MHTP2(s) + (norm(x-x_MHTP2) < tol_suc*norm_x) ;
                    
                    tic ; 
                    [x_NHTP,i_NHTP] = NHTP(A,y,s) ; 
                    t_NHTP = toc ;     
                    Time_NHTP(s) = Time_NHTP(s) + t_NHTP ;   
                    Iter_NHTP(s) = Iter_NHTP(s)+ i_NHTP ;  
                    Res_NHTP(s) = Res_NHTP(s) + (norm(x-x_NHTP) < tol_suc*norm_x) ; 
                    
                    tic ; 
                    [x_MIHT,i_MIHT] = MHTP(A,y,s,0) ; 
                    t_MIHT = toc ;     
                    Time_MIHT(s) = Time_MIHT(s) + t_MIHT ;   
                    Iter_MIHT(s) = Iter_MIHT(s)+ i_MIHT ;  
                    Res_MIHT(s) = Res_MIHT(s) + (norm(x-x_MIHT) < tol_suc*norm_x) ; 
                    
                    tic ;  
                    x_BP = BasisPursuitLP(A,y) ; 
                    t_BP = toc ; 
                    Time_BP(s) = Time_BP(s) + t_BP ; 
                    Res_BP(s) = Res_BP(s) + (norm(x-x_BP) < tol_suc*norm_x) ; 
                    
                    tic ; 
                    [x_MHTP1,i_MHTP1,n_MHTP1] = MHTP(A,y,s,1,k) ; 
                    t_MHTP1 = toc ;     
                    Time_MHTP1(s) = Time_MHTP1(s) + t_MHTP1 ;   
                    Iter_MHTP1(s) = Iter_MHTP1(s) + i_MHTP1 ; 
                    nIt_MHTP1(s) = nIt_MHTP1(s) + n_MHTP1 ;
                    Res_MHTP1(s) = Res_MHTP1(s) + (norm(x-x_MHTP1) < tol_suc*norm_x) ;
                    
                end
          end
end    

t2c = sum(Time_MHTP2)+sum(Time_NHTP)+sum(Time_MIHT)+sum(Time_BP)+sum(Time_MHTP1) ; 

save('Exp2c.mat')


%% Visualization of the results of Experiment 2c
try load('Exp2c.mat')
catch
    load('Exp2c_default.mat')
end

S_range = [1:s_max] ; 
figure
        plot(S_range, Res_MHTP2/n_Meas/n_Vec,'k:x',...
            S_range, Res_NHTP/n_Meas/n_Vec,'b--o',...
            S_range, Res_MIHT/n_Meas/n_Vec,'r-d',...
            S_range, Res_BP/n_Meas/n_Vec,'g-.+',...
            S_range, Res_MHTP1/n_Meas/n_Vec, 'y--v') ;
        legend('MHTP2','NHTP','MIHT','BP','MHTP1','Location','northeast');
        title(strcat('Recovery success (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Linear'),'FontSize',16);
    xlabel('Sparsity (s)','FontSize',22);
    ylabel('Frequency of success','FontSize',22); 