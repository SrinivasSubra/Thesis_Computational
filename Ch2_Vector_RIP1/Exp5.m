%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reproducible file accompanying chapter 2 of the 
% Thesis "Iterative algorithms for sparse and low rank recovery"
% by Srinivas Subramanian  
% Chapter 2: ITERATIVE ALGORITHMS FOR SPARSE RECOVERY FROM MEASUREMENT MATRICES SATISFYING AN l1 RESTRICTED ISOMETRY PROPERTY 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 5a: Robustness on Rademacher vectors ("flat")
% Reconstruction error as a function of the l1-norm of the measurement error
% Decay of number of correctly captured indices as a function of the l1-norm of the measurement error 

clear variables; clc;

% define the problem sizes
N = 1500 ; 
m = 500 ; 
s = 25 ; 
% set numbers of random trials 
n_Meas = 20 ;    % number of measurement matrices
n_Vec = 10 ;    % number of sparse vectors per measurement matrix

norme_max = 7 ;
t_max = 100 ;

Err_MHTP2 = zeros(1,t_max) ;  
Cind_MHTP2 = zeros(1,t_max) ;

Err_NHTP = zeros(1,t_max) ; 
Cind_NHTP = zeros(1,t_max) ;

Err_MIHT = zeros(1,t_max) ; 
Cind_MIHT = zeros(1,t_max) ;

Err_MHTP1 = zeros(1,t_max) ;  
Cind_MHTP1 = zeros(1,t_max) ;

k = 50 ; 

tic ;
for meas = 1:n_Meas
    % tracks experiment progress
    fprintf('Number of measurement schemes tested = %d \n',meas-1);
   % define random A 
    A = randlap([m,N],1)/m ;
    for vect = 1:n_Vec
         % the sparse x to be recovered 
        x = zeros(N,1) ;
        supp = sort(randperm(N,s)) ;       
        x(supp) = sign(randn(s,1)) ; % x = sgn(randn)
        xs_star = 1 ; % smallest non-zero value of x     
        for t=1:t_max
            norme = (t/t_max)*norme_max;  % eta
            e = randn(m,1) ;
            e = norme/norm(e,1)*e*xs_star ;
            y = A*x + e ; 
            % perform the reconstructions and record results 
                
                    [x_MHTP2,~,~,S_MHTP2] = MHTP(A,y,s,2) ; 
                    er_MHTP2 = norm(x-x_MHTP2)/xs_star ; 
                    c_MHTP2 = length(intersect(S_MHTP2,supp)) ;
                    Err_MHTP2(t) = Err_MHTP2(t) + er_MHTP2 ;   
                    Cind_MHTP2(t) = Cind_MHTP2(t) + c_MHTP2 ; 
                   
                    [x_NHTP,~,S_NHTP] = NHTP(A,y,s) ; 
                    er_NHTP = norm(x-x_NHTP)/xs_star ; 
                    c_NHTP = length(intersect(S_NHTP,supp)) ;
                    Err_NHTP(t) = Err_NHTP(t) + er_NHTP ;   
                    Cind_NHTP(t) = Cind_NHTP(t) + c_NHTP ; 
                    
                    [x_MIHT,~,~,S_MIHT] = MHTP(A,y,s,0) ; 
                    er_MIHT = norm(x-x_MIHT)/xs_star ; 
                    c_MIHT = length(intersect(S_MIHT,supp)) ;
                    Err_MIHT(t) = Err_MIHT(t) + er_MIHT ;   
                    Cind_MIHT(t) = Cind_MIHT(t) + c_MIHT ;
                         
                    [x_MHTP1,~,~,S_MHTP1] = MHTP(A,y,s,1,k) ; 
                    er_MHTP1 = norm(x-x_MHTP1)/xs_star ; 
                    c_MHTP1 = length(intersect(S_MHTP1,supp)) ;
                    Err_MHTP1(t) = Err_MHTP1(t) + er_MHTP1 ;   
                    Cind_MHTP1(t) = Cind_MHTP1(t) + c_MHTP1 ;                    
        end
    end 
end
Err_MHTP2 = Err_MHTP2/n_Meas/n_Vec ;
Cind_MHTP2 = Cind_MHTP2/n_Meas/n_Vec ;

Err_NHTP = Err_NHTP/n_Meas/n_Vec ;
Cind_NHTP = Cind_NHTP/n_Meas/n_Vec ;

Err_MIHT = Err_MIHT/n_Meas/n_Vec ;
Cind_MIHT = Cind_MIHT/n_Meas/n_Vec ;

Err_MHTP1 = Err_MHTP1/n_Meas/n_Vec ;
Cind_MHTP1 = Cind_MHTP1/n_Meas/n_Vec ;

etas = (1:t_max)*norme_max/t_max ; 

t5a = toc;
save('Exp5a.mat');

%% Visualization of the results of Experiment 5a

try load('Exp5a.mat')
catch
    load('Exp5a_default.mat')
end

figure
        plot(etas,Err_MHTP2,'k:x',...
            etas, Err_NHTP,'b--o',...
            etas, Err_MIHT,'r-d',...
            etas, Err_MHTP1, 'g--v') ;
        legend('MHTP2','NHTP','MIHT','MHTP1','Location','southeast');
        title(strcat('Recovery error (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m), ', s =',num2str(s),'\newline','x shape: Flat'),'FontSize',16);
     xlabel(' Measurement error ','FontSize',22);
     ylabel(' Normalized reconstruction error ','FontSize',22); 

figure
        plot(etas,Cind_MHTP2,'k:x',...
            etas, Cind_NHTP,'b--o',...
            etas, Cind_MIHT,'r-d',...
            etas, Cind_MHTP1, 'g--v') ;
        legend('MHTP2','NHTP','MIHT','MHTP1','Location','northeast');
        title(strcat('Number of correct indices captured (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m), ', s =',num2str(s),'\newline','x shape: Flat'),'FontSize',16);
      xlabel(' Measurement error ','FontSize',22);
     ylabel(' Number of correct indices captured ','FontSize',22); 

     
     %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 5b: Robustness on vectors with linear decreasing entries  
% Reconstruction error as a function of the l1-norm of the measurement error
% Decay of number of correctly captured indices as a function of the l1-norm of the measurement error 

clear variables; clc;

% define the problem sizes
N = 1500 ; 
m = 500 ; 
s = 25 ; 
% set numbers of random trials 
n_Meas = 20 ;    % number of measurement matrices
n_Vec = 10 ;    % number of sparse vectors per measurement matrix

norme_max = 0.5 ;
t_max = 100 ;

Err_MHTP2 = zeros(1,t_max) ;  
Cind_MHTP2 = zeros(1,t_max) ;

Err_NHTP = zeros(1,t_max) ; 
Cind_NHTP = zeros(1,t_max) ;

Err_MIHT = zeros(1,t_max) ; 
Cind_MIHT = zeros(1,t_max) ;

Err_MHTP1 = zeros(1,t_max) ;  
Cind_MHTP1 = zeros(1,t_max) ;

k = 50 ; 

tic ;
for meas = 1:n_Meas
    % tracks experiment progress
    fprintf('Number of measurement schemes tested = %d \n',meas-1);
   % define random A 
    A = randlap([m,N],1)/m ;
    for vect = 1:n_Vec
         % the sparse x to be recovered 
        x = zeros(N,1) ;
        supp = sort(randperm(N,s)) ;       
        x(supp) = 1+(1-[1:s])/s ; 
        xs_star = 1/s ; % smallest non-zero value of x     
        for t=1:t_max
            norme = (t/t_max)*norme_max ;  % eta
            e = randn(m,1) ;
            e = norme/norm(e,1)*e ;
            y = A*x + e ; 
            % perform the reconstructions and record results 
                
                    [x_MHTP2,~,~,S_MHTP2] = MHTP(A,y,s,2) ; 
                    er_MHTP2 = norm(x-x_MHTP2)/xs_star ; 
                    c_MHTP2 = length(intersect(S_MHTP2,supp)) ;
                    Err_MHTP2(t) = Err_MHTP2(t) + er_MHTP2 ;   
                    Cind_MHTP2(t) = Cind_MHTP2(t) + c_MHTP2 ; 
                   
                    [x_NHTP,~,S_NHTP] = NHTP(A,y,s) ; 
                    er_NHTP = norm(x-x_NHTP)/xs_star ; 
                    c_NHTP = length(intersect(S_NHTP,supp)) ;
                    Err_NHTP(t) = Err_NHTP(t) + er_NHTP ;   
                    Cind_NHTP(t) = Cind_NHTP(t) + c_NHTP ; 
                    
                    [x_MIHT,~,~,S_MIHT] = MHTP(A,y,s,0) ; 
                    er_MIHT = norm(x-x_MIHT)/xs_star ; 
                    c_MIHT = length(intersect(S_MIHT,supp)) ;
                    Err_MIHT(t) = Err_MIHT(t) + er_MIHT ;   
                    Cind_MIHT(t) = Cind_MIHT(t) + c_MIHT ;
                         
                    [x_MHTP1,~,~,S_MHTP1] = MHTP(A,y,s,1,k) ; 
                    er_MHTP1 = norm(x-x_MHTP1)/xs_star ; 
                    c_MHTP1 = length(intersect(S_MHTP1,supp)) ;
                    Err_MHTP1(t) = Err_MHTP1(t) + er_MHTP1 ;   
                    Cind_MHTP1(t) = Cind_MHTP1(t) + c_MHTP1 ;                    
        end
    end 
end
Err_MHTP2 = Err_MHTP2/n_Meas/n_Vec ;
Cind_MHTP2 = Cind_MHTP2/n_Meas/n_Vec ;

Err_NHTP = Err_NHTP/n_Meas/n_Vec ;
Cind_NHTP = Cind_NHTP/n_Meas/n_Vec ;

Err_MIHT = Err_MIHT/n_Meas/n_Vec ;
Cind_MIHT = Cind_MIHT/n_Meas/n_Vec ;

Err_MHTP1 = Err_MHTP1/n_Meas/n_Vec ;
Cind_MHTP1 = Cind_MHTP1/n_Meas/n_Vec ;

etas = (1:t_max)*norme_max/t_max ; 

t5b = toc;
save('Exp5b.mat');

%% Visualization of the results of Experiment 5b

try load('Exp5b.mat')
catch
    load('Exp5b_default.mat')
end

figure
        plot(etas,Err_MHTP2,'k:x',...
            etas, Err_NHTP,'b--o',...
            etas, Err_MIHT,'r-d',...
            etas, Err_MHTP1, 'g--v') ;
        legend('MHTP2','NHTP','MIHT','MHTP1','Location','southeast');
        title(strcat('Recovery error (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m), ', s =',num2str(s),'\newline','x shape: Linear'),'FontSize',16);
     xlabel(' Measurement error ','FontSize',22);
     ylabel(' Normalized reconstruction error ','FontSize',22); 
figure
        plot(etas,Cind_MHTP2,'k:x',...
            etas, Cind_NHTP,'b--o',...
            etas, Cind_MIHT,'r-d',...
            etas, Cind_MHTP1, 'g--v') ;
        legend('MHTP2','NHTP','MIHT','MHTP1','Location','northeast');
        title(strcat('Number of correct indices captured (averaged over', 32, num2str(n_Meas*n_Vec),' trials)',...
            '\newline','N=',num2str(N), ...
            ', m=',num2str(m), ', s =',num2str(s),'\newline','x shape: Linear'),'FontSize',16);
      xlabel(' Measurement error ','FontSize',22);
     ylabel(' Number of correct indices captured ','FontSize',22); 
