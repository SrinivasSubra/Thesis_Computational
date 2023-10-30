%%
% MIHT.m
% Perform low-rank recovery from rank-one measurements
% via Modified Iterative Hard Thresholding 
% or Modified Hard Thresholding Pursuit
%
% Implements the MIHT/MHTP algorithms analyzed in Chapter 3 of the thesis
% Thesis "Iterative algorithms for sparse and low-rank recovery from atypical measurements"
% by Srinivas Subramanian.
% I.e., tries to recover a matrix X of rank <=r acquired via y = A(X)+e
% from the iterations X_{n+1} = H_s[ X_n + mu_n*H_t[A'(sgn(y-A(X_n)))]]
% if htp flag = 1, this is followed by additional least squares projection
% steps resulting in Modified Hard Thresholding Pursuit algorithm.
% Usage: [Xn,n,Rres] = MIHT(A,B,y,r,...)
%
% A: a matrix with columns a_1,...,a_m
% B: a matrix with columns b_1,...,b_m
% y: the measurement vector with entries a_i'*X*b_i + e_i
% r: an (over)estimation of the rank of the matrix X to be recovered
%
% Other optional inputs:
% s_over_r: outer thresholding parameter, sets s = s_over_r*r (default: 1)
% t_over_r: inner thresholding parameter, sets t = t_over_r*r (default: Inf)
% htp: binary flag to perform the MHTP algorithm (default: 0)
%   if htp=1, MIHT becomes MHTP, i.e, after the IHT step, 
% additional least squares projection steps are performed to obtain the "best fit"
%   if htp=0, no projections are performed, just remains as the MIHT algorithm
% gamma : estimate of Modified Rank-Restricted Isometry Ratio (default: 3)
% itmax: maximal number of iterations
% tol: tolerance for the stopping criterion Rres < tol
% stop: binary flag adding a stopping criterion (default: 1)
%   if stop=1, MIHT stops when a change of maximizer in the stepsize occurs
%   if stop=0, MIHT does not stop when such a change occurs
%
% Xn: a matrix approximating X produced by the MIHT/MHTP algorithm
% n: the number of iterations performed
% Rres: the relative L1-norm of the residual

% Written by Srinivas Subramanian and Simon Foucart 

function [Xn,n,Rres] = MIHT(A,B,y,r,varargin)

% MIHT takes 4 inputs and at most 7 optional inputs
numvarargs = length(varargin);
if numvarargs > 7
    error('MIHT can have at most 7 optional inputs: s_over_r, t_over_r, htp, gamma, itmax, tol, stop');
end
% set defaults for optional inputs:
% s_over_r = 1, so that s = r
% t_over_r = Inf, in order to discard the inner SVD performed by H_t
% htp = 0, MIHT algorithm (no least squares projections)
% gamma = 3, valid for Gaussian rank-one projections
% itmax = 1000
% tol = 1e-4
% stop = 1, so that the stopping criterion nu2 > nu1 is added
optargs = {1, Inf, 0, 3, 1000, 1e-4, 1};
% skip new inputs if they are empty
newVals = cellfun(@(x) ~isempty(x), varargin);
% modify the default inputs with the optional inputs
optargs(newVals) = varargin(newVals);
% place optional arguments in variable names
[s_over_r,t_over_r,htp,gamma,itmax,tol,stop] = optargs{:};

N1 = size(A,1);
N2 = size(B,1);
s = s_over_r*r;
t = t_over_r*r;
if t>=min(N1,N2)
    NoHt = 1;  % flag to discard the inner SVD performed by H_t
else
    NoHt = 0;
end

% initilaization
n = 0 ;
Xn = zeros(N1,N2);
res = y - sum(A.*(Xn*B))';
Rres = norm(res,2)/norm(y,2);

% main loop
while((n < itmax)&&(Rres > tol))
    M = A*diag(sign(res))*B';  % this is the matrix A'(sgn(y-A(X_n)))
    if NoHt
        Ht = M;  % no inner SVD is performed
    else
        [U,S,V] = svds(M,t);
        Ht = U*S*V';
    end
    nu1 = norm(Ht,'fro');
    nu2 = (1/(2*gamma)*norm(sum(A.*(Ht*B)),1)/norm(Ht,'fro'));
    [nu,i] = max([nu1,nu2]);
    if((stop==1)&&(i==2))
        break;  % terminate when the stopping criterion nu2 > nu1 is met
    end
    mu = norm(res,1)/nu^2;
    [U,S,V] = svds(Xn+mu*Ht,s);
    if htp         % if we want MHTP, need to compute best V,U and sigma
        V = (mls(y, S*U'*A, B))'; % computes the best V via least squares
        U = mls(y, A, S*V'*B); % computes the best U via least squares
        C = (B'*V).*(A'*U); % form C to compute best sigma via least squares
        sig = C\y;
        S = diag(sig);
    end
    Xn = U*S*V';
    res = y - sum(A.*(Xn*B))';
    Rres = norm(res,2)/norm(y,2);
    n = n+1 ;
end

end
