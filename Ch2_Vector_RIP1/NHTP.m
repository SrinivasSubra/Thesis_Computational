%%
% NHTP.m 
% Perform sparse recovery from measurements obtained using matrices 
% satisyfing an l1 restricted isometery property, 
% eg: Laplacian random measurements, 
% via Normalized Hard Thresholding Pursuit algorithm of Simon Foucart
%
% Implements NHTP algorithm put forward in the article  
% "Hard thresholding pursuit: an algorithm for compressive sensing" 
% by Simon Foucart.
% % I.e., tries to recover a vector x of sparsity <=s acquired via y = Ax + e
% from the iterations where each iteration involves 
% a hard thresholding step with a varying step size 
% followed by finding a vector with the current support that best fits the
% measurements using l2 minimization 
% See the article for more details. 
% Usage: [x,n,S,stp,Rres] = NHTP(A,y,s,...)
%
% A: measurement matrix with N columns 
% y: measurement vector A*x + e
% s: an (over)estimation of the sparsity of the vector x to be recovered 
% 
% Other optional inputs: 
% Rres_stp: flag to add stopping criterion that checks if the relative
%           residual is less than tolerance, i.e, stops algorithm if Rres < tol (default: 0)   
% itmax: maximal number of iterations
% tol: tolerance for the stopping criterion Rres < tol
%
% x: vector produced by the algorithm that approximates the vector to be recovered  
% n: number of iterations performed
% S: support of the solution vector 
% stp: flag to indicate which stopping criterion was used to stop algorithm
%     if stp = 1, the natural stopping criterion S_{n+1} = S_n was used 
% Rres: the relative l2 norm of the residual 

function [x,n,S,stp,Rres] = NHTP(A,y,s,varargin) 
% NHTP takes 3 inputs and at most 3 optional inputs
numvarargs = length(varargin);
if numvarargs > 3
    error('NHTP can have at most 3 optional inputs: Rres_stp, itmax, tol');
end
% set defaults for optional inputs:
% itmax = 1000
% Rres_stp = 0, so that default stopping condition is S_{n+1} = S_n 
% tol = 1e-6
optargs = {0, 1000, 1e-6};
% skip new inputs if they are empty
newVals = cellfun (@(x) ~isempty(x), varargin);
% modify the default inputs with the optional inputs
optargs (newVals) = varargin (newVals);
% place optional arguments in variable names
[Rres_stp, itmax, tol] = optargs{:};
[~,N] = size(A) ;
% initialization
n = 0 ;
stp = 0 ; 
xn = zeros(s,1) ;
res = y ; 
Rres = 1 ;
mu = 1 ; 
S = 1:s ; 
Sprev = [];
while((n < itmax)&&( ~Rres_stp||(Rres > tol) )) 
    u = mu*A'*res;
    u(S) = u(S) + xn ; % xn is supported on S from the previous iteration
    [~,S] = maxk(u, s, 'ComparisonMethod', 'abs') ; % hard thresholding H_s
    S = sort(S) ; 
                          % compare S_{n+1} with S_n 
    if(isequal(Sprev,S))  % stopping criterion for NHTP 
        stp = 1 ; % stopped flag
        break ; 
    end
    Sprev = S ; 
    AS = A(:,S) ; 
    xn = AS\y ; % ordinary least squares  
    res = y-AS*xn ; 
    Rres = norm(res,2)/norm(y,2) ;
    n = n+1 ;  
    w = AS'*res ;
    mu = (norm(w,2)/norm(AS*w,2))^2 ; %normalized HTP  
    if((n==1)&&(Rres < tol)) % edge case stopping condition for S_1, since there is no Sprev
        break ;              % without this, res can potentially become 0 and give a NaN value of step size mu    
    end
end

x = zeros(N,1) ; 
x(S) = xn ; 

end 

