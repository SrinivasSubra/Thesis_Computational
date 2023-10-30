%%
% MHTP.m
% Perform sparse recovery from measurements obtained using matrices 
% satisyfing an l1 restricted isometery property, 
% eg: Laplacian random measurements, 
% via Modified Hard Thresholding Pursuit with l1 or l2 minimizations 
% or Modified Iterative Hard Thresholding 
%
% Implements the MIHT/MHTP algorithms analyzed in Chapter 2 of the thesis 
% "Iterative algorithms for sparse and low-rank recovery from atypical measurements"
% by Srinivas Subramanian. 
% I.e., tries to recover a vector x of sparsity <=s acquired via y = Ax + e
% from the iterations where each outer iteration involves 
% k steps of modified iterative hard thresholding of order kappa*s, 
% followed by finding a vector with the current support that best fits the
% measurements using  
% l1 minimization if p = 1 
% l2 minimization if p = 2 
% See Chapter 3 of the thesis for more details. 
% MIHT is implemented if p = 0, in which case parameter k is redundant 
% Usage: [x,iter,n,S,stp,Rres] = MHTP(A,y,s,...)
%
% A: measurement matrix with N columns 
% y: measurement vector A*x + e
% s: an (over)estimation of the sparsity of the vector x to be recovered 
% 
% Other optional inputs:
% p: ternary flag that indicates which algorithm to implement (default: 2) 
%     if p = 0, implements the MIHT algorithm 
%     if p = 1, implements the MHTP1 algorithm with l1 minimizations 
%     if p = 2, implements the MHTP2 algorithm with l2 minimizations 
% k: number to indicate how often minimzation is performed for the MHTP
%    algorithms, i.e, performs a minimization once at every kth iteration (default: 1)   
% kappa: outer thresholding parameter, performs H_{kappa*s} thresholding  (default: 1)   
% Rres_stp: flag to add stopping criterion that checks if the relative
%           residual is less than tolerance, i.e, stops algorithm if Rres < tol (default: 0)   
% itmax: maximal number of iterations
% tol: tolerance for the stopping criterion Rres < tol
%
% x: vector produced by the algorithm that approximates the vector to be recovered 
% iter: total number of iterations performed 
% n: number of minimizations performed/number of outer iterations
% S: support of the solution vector 
% stp: flag to indicate which stopping criterion was used to stop algorithm
%     if stp = 1, the natural stopping criterion S_{n+1} = S_n was used 
% Rres: the relative l1 norm of the residual 

function [x, iter, n, S, stp, Rres] = MHTP (A, y, s, varargin)

% MHTP takes 3 inputs and at most 6 optional inputs
numvarargs = length(varargin);
if numvarargs > 6
    error('MHTP can have at most 6 optional inputs: p, k, kappa, Rres_stp, itmax, tol');
end
% set defaults for optional inputs:
% p = 2, for l2 minimization step. 
% k = 1, every step involves a minimization 
% kappa = 1, so thresholding H_s is performed 
% Rres_stp = 0, default stopping criterion for MHTP is  S_{n+1} = S_n
% itmax = 1000
% tol = 1e-6
optargs = {2, 1, 1, 0, 1000, 1e-6};
% skip new inputs if they are empty
newVals = cellfun (@(x) ~isempty(x), varargin);
% modify the default inputs with the optional inputs
optargs (newVals) = varargin (newVals);
% place optional arguments in variable names
[p, k, kappa, Rres_stp, itmax, tol] = optargs{:};

if(~p||(k>1))  % for MIHT, always set the flag as 1 since Rres < tol is the stopping condition
    Rres_stp = 1 ; % k > 1 : check if Rres<tol and solution found before S_{n+1}=S_n, since 
end                % without this, res can potentially become 0 and give a NaN value of step size nu
% initilaization
[~, N] = size (A) ;
xl = zeros (kappa*s,1) ;
n = 0 ;
r = (2*kappa + 1)*s ; 
l = 1 ; % l cycles from 1 to k, for MHTP do minimization step everytime l = k
res = y ; % residual 
Rres = 1 ; 
iter = 0 ; % total number of iterations 
stp = 0 ; 
S = 1:s; 
Sprev = []; 
% main loop
while((iter < itmax)&&( ~Rres_stp||(Rres > tol) )) % add Res<tol as a stopping condition if specified
    
    u = A'*sign(res) ; 
    w = maxk(u, r, 'ComparisonMethod', 'abs') ;
    nu = norm(res,1)/norm(w)^2 ; % step size 
    u = nu*u ; 
    u(S) = u(S) + xl ; % xl is supported on S from the previous iteration
    [~,S] = maxk(u, kappa*s, 'ComparisonMethod', 'abs') ; % hard thresholding H_{kappa*s}
    S = sort(S) ; 
    xl = u(S) ; 
    if((p)&&(l==k))  % at every kth iteration do minimization for MHTP1 if p = 1, MHTP2 if p = 2, and skips if p = 0 for MIHT  
        if(p==2) 
            xl = A(:,S)\y ;  % l2 least squares if p = 2 
        else  
            xl = l1approxprog (A(:,S), y) ;  % l1 approximation if p = 1 
        end  
                              % compare S_{n+1} with S_n, n>=1   
        if(isequal(Sprev,S))  % stopping criterion for MHTP 
            stp = 1 ; % stopped flag
            break ; 
        end
        
        Sprev = S ; 
        n = n + 1 ; l = 0 ; % increment n, reset l 
    end
    
    res = y - A(:,S)*xl ; 
    Rres = norm(res,2)/norm(y,2) ; 
    l = l + 1 ; 
    iter = iter + 1 ; 
    
    if(( (n==1) )&&(Rres < tol))  % n == 1 : edge case stopping condition for S_1, since there is no Sprev
        break ;                   % without this, res can potentially become 0 and give a NaN value of step size nu         
    end                                  
    
end
x = zeros(N,1) ;       
x(S) = xl ; 

end