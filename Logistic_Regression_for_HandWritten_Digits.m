clear all;
close all;
clc; 
load mnist_49_3000 [d,n] = size(x); 
lambda = 10;
% convert label space from {-1, +1} to {0, 1} as described in notes. y = (y + 1) /2;
ntr = 2000; nts = n - ntr;
% append vector of 1 for the bias term. 
xtilde = [ones(1,n); x];
% split data into training/testing set 
xtr = xtilde(:,1:ntr); 
xts = xtilde(:,ntr+1:n); 
ytr = y(1:ntr); 
yts = y(ntr+1:n);
% sigmoid/logistic function and its derivative function
g = @(t) 1./(1+exp(-t)); 
dg = @(t) exp(-t)./(1+exp(-t)).^2;
% initialize with the zero vector. +1 for the bias term theta = zeros(d+1,1);
iter = 0; 
for iter = 1:50 % objective function value (regularized negative log-likelihood) 
    objval_old = -ytr*log(g(theta'*xtr)+eps)'+... 
        (ytr-1)*log(1-g(theta'*xtr)+eps)' + lambda*norm(theta)^2;
% gradient of the objective 
dl = xtr * ( g(theta'*xtr) - ytr )' + 2*lambda*theta;
% hessian matrix 
H = xtr*diag(dg(theta'*xtr))*xtr' + 2*lambda*eye(d+1); % you can also use the loop below to create the Hessian
% H = zeros(d+1,d+1); 
% for i = 1:ntr 
% H = H + xtr(:,i) * xtr(:,i)’ * dg(theta’*xtr(:,i)); 
% end 
% H = H + 2*lambda*eye(d+1);
% newton’s update 
theta = theta - H\dl;
% new objective function value 
objval = -ytr*log(g(theta'*xtr)+eps)' + ... 
    (ytr-1)*log(1-g(theta'*xtr)+eps)' + lambda*norm(theta)^2;
% (c) termination criteria: relative change of the objective function 
if abs( (objval - objval_old)/objval_old) < 1e-6
break 
end
end
% logistic regression estimate 
yLR = (theta'*xts) >= 0; LRtesterror = sum(yLR ~= yts)/nts;
% (a) 
fprintf('The test error of the logistic regression classifier is %3.1f%%\n',... 
    LRtesterror*100) 
% (b) 
fprintf('The value of the regularized log-likelihood at the optimum is %3.2f\n',... 
    objval)
% the misclassified image indices 
indmiss = find(yLR ~= yts);
t = theta'*xts(:,indmiss);
% the indices the logistic regression classifier was most confident of 
[tsort, tsortind] = sort(abs(t), 'descend'); 
figure
for i = 1:20 
    ind = indmiss(tsortind(i)); 
    subplot(4,5,i) 
    imagesc( reshape( xts(2:d+1,ind), [sqrt(d), sqrt(d)])' ) 
    colormap gray 
    axis off square 
    if yts(ind) == 1
        str = '9'; 
    else
        str = '4';
    end
    title(str) 
end