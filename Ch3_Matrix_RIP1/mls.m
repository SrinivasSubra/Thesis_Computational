% Matrix least squares 
% solves argmin ||y - Alin(X)||_2 = || y - C*vec(X) ||_2
% y is of size m, A is N1-by-m, B is N2-by-m 
% Alin is the linear operator for rank one projections defined by the
% matrices A and B. 
function X = mls(y,A,B) 
[N1,~] = size(A); 
[N2,~] = size(B); 
D = repelem(B', 1, N1); % repeats each column of B' N1 times 
E = repmat(A', 1, N2); % repeats entire A' N2 times [A',A',...A'] 
C = D.*E; % form E,D as a way to avoid loops, c = vec(ab')' or C(i,:) = (vec(A(:,i)*B(:,i)'))'
X = reshape(C\y, N1, N2); %least squares solution reshaped into a matrix 
end
