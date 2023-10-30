% Finds the best approximation to b in l1 norm using linprog
% x = argmin ||b-Az||_1  over z 
% A is m-by-n and b is n-by-1, and m>=n
% LP formulation: with r = Az - b, r+ = u, r- = v, r = u - v, |r_i| = u_i + v_i
% we have Az - b = u - v and ||b-Az||_1 = ones'u + ones'v, so we solve 
% minimize zeros'z + ones'u + ones'v 
% subject to  Az - u + v = b, u>=0, v>=0 
% Usage: x = l1approxprog(A,b) 

% For our experiment problem sizes, linprog seemed to be faster than CVX 

function x = l1approxprog(A,b)

m = length(b) ; 
n = size(A,2) ; 
f    = [ zeros(n,1); ones(m,1);  ones(m,1)  ] ;
Aeq  = [ A,          -eye(m),    +eye(m)    ] ;
lb   = [ -Inf(n,1);  zeros(m,1); zeros(m,1) ] ;
Options = optimoptions('linprog', 'Display', 'off');
xzz  = linprog(f,[],[],Aeq,b,lb,[], Options) ;
x = xzz(1:n,:) ; 

end

