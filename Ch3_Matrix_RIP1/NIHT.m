%%
% NIHT.m
% Perform low-rank recovery from rank-one measurements
% via Normalized Iterative Hard Thresholding of Tanner and Wei
%
% Implements the NIHT algorithm put forward in the article
% "Normalized iterative hard thresholding for matrix completion"
% by J. Tanner and K. Wei
% I.e., tries to recover a matrix X of rank <=r acquired via y = A(X)+e
% from the iterations X_{n+1} = H_r[ X_n + mu_n*A'(y-A(X_n))]
%
% Usage: [Xn,n,Rres] = NIHT(A,B,y,r,...)
%
% A: a matrix with columns a_1,...,a_m
% B: a matrix with columns b_1,...,b_m
% y: the measurement vector with entries a_i'*X*b_i + e_i
% r: an (over)estimation of the rank of the matrix X to be recovered
%
% Other optional inputs:
% itmax: maximal number of iterations
% tol: tolerance for the stopping criterion Rres < tol
%
% Xn: a matrix approximating X produced by the NIHT algorithm
% n: the number of iterations performed
% Rres: the relative L2-norm of the residual

% Written by Srinivas Subramanian and Simon Foucart



function [Xn,n,Rres] = NIHT(A,B,y,r,varargin)

% NIHT takes 4 inputs and at most 2 optional inputs
numvarargs = length(varargin);
if numvarargs > 2
    error('NIHT can have at most 2 optional inputs: itmax,tol');
end
% set defaults for optional inputs:
optargs = {1000, 1e-4};
% skip any new inputs if they are empty
newVals = cellfun(@(x) ~isempty(x), varargin);
% modify the default inputs with the optional inputs
optargs(newVals) = varargin(newVals);
% place optional arguments in variable names
[itmax, tol] = optargs{:};

% initialization
n = 0;
M = A*diag(y)*B';
[U,S,V] = svds(M,r);
Xn = U*S*V';
res = y - sum(A.*(Xn*B))';
Rres = norm(res,2)/norm(y,2); 

% main loop
while((n < itmax)&&(Rres > tol))
    M = A*diag(res)*B';    % this is the matrix A'(y-A(X_n))
    [U,~,~] = svds(Xn,r);  % left singular vectors needed to define mu_n
    mu = ( norm((U*U'*M),'fro')/norm(sum(A.*(U*U'*M*B))) )^2;
    [U,S,V] = svds(Xn+mu*M,r);
    Xn = U*S*V';
    res = y - sum(A.*(Xn*B))';
    Rres = norm(res,2)/norm(y,2);
    n = n+1;
end

end