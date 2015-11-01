function [] = linear_and_robust_linear_regression() 
close all;
n = 200; 
rng(0); 
x = rand(n,1); 
z = zeros(n,1);
k = n*0.5; 
rp = randperm(n); 
outlier_subset = rp(1:k); 
z(outlier_subset)=1; % outliers 
y = (1-z).*(10*x + 5 + randn(n,1)) + z.*(20 - 20*x + 10*randn(n,1));
scatter(x,y,'b') 
hold on 
t = 0:0.01:1; 
plot(t,10*t+5,'k');
c = ones(size(x));
[w_ols,b_ols] = wls(x,y,c); 
plot(t, w_ols*t + b_ols, 'g--');
w_rob = 0; 
b_rob = 0; 
rho = @(r) sqrt(1+r.^2)-1; 
obj_new = (1/n)*sum(rho(y - w_rob'*x - b_rob));
for i=1:100 obj_old = obj_new;
    c = 1./sqrt(1 + (y - w_rob'*x - b_rob).^2);
    [w_rob, b_rob] = wls(x,y,c); 
    obj_new = (1/n)*sum(rho(y - w_rob'*x - b_rob));
if obj_old < obj_new + 1e-6
break 
end
end
plot(t, w_rob*t + b_rob, 'r:');
legend('data','true line','least squares','robust')
fprintf(['Parameters esimated from least squares error regression: ',...
'(w,b)= (%3.2f,%3.2f)\n'], w_ols,b_ols) 
fprintf(['Parameters esimated from robust: ',... 
    '(w,b)= (%3.2f,%3.2f)\n'], w_rob,b_rob)
function [w,b] = wls(x,y,c) 
% weighted least squares 
[n,d] = size(x); 
X = [ones(n,1) x];
C = diag(c); 
theta = (X'*C*X) \ X'*C*y; 
b = theta(1); 
w = theta(2); 
end
end