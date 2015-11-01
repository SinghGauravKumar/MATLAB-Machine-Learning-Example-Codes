function [] = subgradient_for_nuclear_data() 
close all;
load nuclear 
n = size(x,2); % display data
figure 
hold on 
scatter(x(1,y==-1),x(2,y==-1),'b') 
scatter(x(1,y==1),x(2,y==1),'rx')
lambda = 0.001;
%=================% % subgradient algorithm %========================%
rng(0); 
theta = [0 0 0]'; 
obj = (1/n)*sum(max(0,1-y.*(theta(1:2)'*x + theta(3)))) + ...
(lambda/2)*(norm(theta(1:2))^2);
for j = 1:40
theta_old = theta; 
u = 0; 
for i=1:n 
    u = u + subg(theta(1:2),theta(3), x(:,i), y(i), lambda, n); 
end
theta = theta - (100/j)*u; 
new_obj = (1/n)*sum(max(0,1-y.*(theta(1:2)'*x + theta(3)))) + ...
(lambda/2)*(norm(theta(1:2))^2);
obj = [obj new_obj];
end
% plot line 
t = 0:0.01:8; 
l = -(theta(1)*t/theta(2) + theta(3)/theta(2)); 
plot(t,l);
figure 
plot(obj)
function [u] = subg(w,b,x,y,lambda,n)
if ((1 - y*(w'*x + b)) > 0) 
    u = [-(1/n)*(y*x - lambda*w);-(1/n)*y];
else
    u = [(1/n)*lambda*w; 0]; 
end
