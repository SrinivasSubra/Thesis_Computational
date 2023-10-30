%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reproducible file accompanying chapter 2 of the 
% Thesis "Iterative algorithms for sparse and low rank recovery"
% by Srinivas Subramanian  
% Chapter 2: ITERATIVE ALGORITHMS FOR SPARSE RECOVERY FROM MEASUREMENT MATRICES SATISFYING AN l1 RESTRICTED ISOMETRY PROPERTY 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 3a:  Performance comparison for several algorithms, with Laplacian A, Gaussian x 

clear variables; clc;

% define the problem sizes
N = 1500 ; 
m = 500 ; 

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 10 ;    % number of measurement matrices
n_Vec = 10 ;    % number of sparse vectors per measurement matrix
tol_suc = 1e-5 ; 

s_min = 1 ; 
s_inc = 4 ; 
s_max = 200 ; % max value of s 
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
                    [x_MHTP1,i_MHTP1,n_MHTP1] = MHTP(A,y,s,1,k) ; 
                    t_MHTP1 = toc ;     
                    Time_MHTP1(t) = Time_MHTP1(t) + t_MHTP1 ;   
                    Iter_MHTP1(t) = Iter_MHTP1(t) + i_MHTP1 ; 
                    nIt_MHTP1(t) = nIt_MHTP1(t) + n_MHTP1 ;
                    Res_MHTP1(t) = Res_MHTP1(t) + (norm(x-x_MHTP1) < tol_suc*norm_x) ;
                    
                end
          end
end    

t3a = sum(Time_MHTP2)+sum(Time_NHTP)+sum(Time_MIHT)+sum(Time_MHTP1) ; 

save('Exp3a.mat')

%% Visualization of the results of Experiment 3a

try load('Exp3a.mat')
catch
    load('Exp3a_default.mat')
end

figure
        plot(s_range, Res_MHTP2/n_Meas/n_Vec,'k:x',...
            s_range, Res_NHTP/n_Meas/n_Vec,'b--o',...
            s_range, Res_MIHT/n_Meas/n_Vec,'r-d',...
            s_range, Res_MHTP1/n_Meas/n_Vec, 'g--v') ;
        legend('MHTP2','NHTP','MIHT','MHTP1','Location','southwest');
        title(strcat('Recovery success (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Gaussian'),'FontSize',16);
    xlabel('Sparsity (s)','FontSize',22);
    ylabel('Frequency of success','FontSize',22); 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 3b:  Performance comparison for several algorithms, with Laplacian A, Rademacher x 


clear variables; clc;

% define the problem sizes
N = 1500 ; 
m = 500 ; 

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 10 ;    % number of measurement matrices
n_Vec = 10 ;    % number of sparse vectors per measurement matrix
tol_suc = 1e-5 ; 

s_min = 1 ; 
s_inc = 4 ; 
s_max = 200 ; % max value of s 
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
                    [x_MHTP1,i_MHTP1,n_MHTP1] = MHTP(A,y,s,1,k) ; 
                    t_MHTP1 = toc ;     
                    Time_MHTP1(t) = Time_MHTP1(t) + t_MHTP1 ;   
                    Iter_MHTP1(t) = Iter_MHTP1(t) + i_MHTP1 ; 
                    nIt_MHTP1(t) = nIt_MHTP1(t) + n_MHTP1 ;
                    Res_MHTP1(t) = Res_MHTP1(t) + (norm(x-x_MHTP1) < tol_suc*norm_x) ;
                    
                end
          end
end    

t3b = sum(Time_MHTP2)+sum(Time_NHTP)+sum(Time_MIHT)+sum(Time_MHTP1) ; 

save('Exp3b.mat')


%% Visualization of the results of Experiment 3b

try load('Exp3b.mat')
catch
    load('Exp3b_default.mat')
end

figure
        plot(s_range, Res_MHTP2/n_Meas/n_Vec,'k:x',...
            s_range, Res_NHTP/n_Meas/n_Vec,'b--o',...
            s_range, Res_MIHT/n_Meas/n_Vec,'r-d',...
            s_range, Res_MHTP1/n_Meas/n_Vec, 'g--v') ;
        legend('MHTP2','NHTP','MIHT','MHTP1','Location','southwest');
        title(strcat('Recovery success (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Flat'),'FontSize',16);
    xlabel('Sparsity (s)','FontSize',22);
    ylabel('Frequency of success','FontSize',22); 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 3c:  Performance comparison for several algorithms, with Laplacian A, linear decreasing entries x 


clear variables; clc;

% define the problem sizes
N = 1500 ; 
m = 500 ; 

% set numbers of random trials, success tolerance, matrix to store results
n_Meas = 10 ;    % number of measurement matrices
n_Vec = 10 ;    % number of sparse vectors per measurement matrix
tol_suc = 1e-5 ; 

s_min = 1 ; 
s_inc = 4 ; 
s_max = 200 ; % max value of s 
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
                    [x_MHTP1,i_MHTP1,n_MHTP1] = MHTP(A,y,s,1,k) ; 
                    t_MHTP1 = toc ;     
                    Time_MHTP1(t) = Time_MHTP1(t) + t_MHTP1 ;   
                    Iter_MHTP1(t) = Iter_MHTP1(t) + i_MHTP1 ; 
                    nIt_MHTP1(t) = nIt_MHTP1(t) + n_MHTP1 ;
                    Res_MHTP1(t) = Res_MHTP1(t) + (norm(x-x_MHTP1) < tol_suc*norm_x) ;
                    
                end
          end
end    


t3c = sum(Time_MHTP2)+sum(Time_NHTP)+sum(Time_MIHT)+sum(Time_MHTP1) ; 

save('Exp3c.mat')


%% Visualization of the results of Experiment 3c
try load('Exp3c.mat')
catch
    load('Exp3c_default.mat')
end
 
figure
        plot(s_range, Res_MHTP2/n_Meas/n_Vec,'k:x',...
            s_range, Res_NHTP/n_Meas/n_Vec,'b--o',...
            s_range, Res_MIHT/n_Meas/n_Vec,'r-d',...
            s_range, Res_MHTP1/n_Meas/n_Vec, 'g--v') ;
        legend('MHTP2','NHTP','MIHT','MHTP1','Location','southwest');
        title(strcat('Recovery success (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m),'\newline','x shape: Linear'),'FontSize',16);
    xlabel('Sparsity (s)','FontSize',22);
    ylabel('Frequency of success','FontSize',22); 