% Finds the best approximation to b in l1 norm using CVX
% x = argmin ||b-Az||_1  over z
% Specify cvx_solver as gurobi or mosek for a faster method  
% Usage: x = l1approx(A,b) 

% This was not used for the experiments since linprog was faster than CVX
% for our experiment problem dimensions. May be better for larger regimes

function x = l1approx(A,b)
[~,n] = size(A);
cvx_begin quiet
    variable x(n)
    minimize( norm(A*x-b,1) )
cvx_end
end

