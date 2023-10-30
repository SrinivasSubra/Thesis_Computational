% Solve basis pursuit using CVX. 
% Specify cvx_solver as gurobi or mosek for a faster method  
% Precision set as low for being faster  
% Usage: x = BasisPursuitcvx(A,b) 

% This seems to be generally faster than basis pursuit using linprog for larger dimensions like N > 1000. 
% For small s, linprog still seems to be faster though 

function x = BasisPursuitcvx(A,b)

[~,n] = size(A);
cvx_begin quiet
cvx_precision('low')
    variable x(n)
    minimize( norm(x,1) )
    subject to 
    A*x == b;
cvx_end

end
