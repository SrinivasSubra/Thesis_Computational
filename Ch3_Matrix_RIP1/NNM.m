%%
% NNM.m
% Perform low-rank recovery from rank-one measurements
% via nuclear norm minimization
%
% Tries to recover a low-rank matrix X acquired via y = A(X)+e
% by solving the minimization program
% min ||Z||_* subject to A(Z) = y
% The implementation is done using CVX
%
% Usage: Xstar = NIHT(A,B,y)
%
% A: a matrix with columns a_1,...,a_m
% B: a matrix with columns b_1,...,b_m
% y: the measurement vector with entries a_i'*X*b_i + e_i
% 
% Xstar: a miminizer of the above program, designed to approximate X  

% Written by Srinivas Subramanian and Simon Foucart 


function Xstar = NNM(A,B,y)

N1 = size(A,1);
N2 = size(B,1); 

cvx_begin
cvx_precision('low')
variable Xstar(N1,N2);

minimize norm_nuc(Xstar);

subject to 
sum(A.*(Xstar*B)) == y';

cvx_end

end
