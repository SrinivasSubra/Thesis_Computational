% LP version of basis pursuit solved using linprog 
% See Pg.63, 3.1 Optimization methods, "A Mathematical Introduction to Compressive Sensing"
% by Simon Foucart for the formulation 
% Usage: x = BasisPursuitLP(A,b) 

% This seems to be generally faster than basis pursuit using CVX for small dimensions, i.e, N <=1000
% For N > 1000 with small s, linprog still seems to be faster though 

function x = BasisPursuitLP(A, b )

n = size(A, 2);
f = ones(2*n,1) ;
Aeq = [A, -A] ;
beq = b ;
lb = zeros(2*n,1);
ub = inf(2*n,1);

Options = optimoptions('linprog', 'Display', 'off');

zpm = linprog(f, [], [], Aeq, beq, lb, ub, Options);

x = zpm(1:n) - zpm(n+1:end);


end