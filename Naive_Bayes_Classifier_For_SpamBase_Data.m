clear all; 
clc; 
z = dlmread('spambase.data',','); 
rand('state', 0); 
rp = randperm(size(z,1)); 
z = z(rp,:); 
x = z(:,1:end-1); 
y = z(:,end);
[n,d] = size(x);
ntrain = 2000; 
xtrain = x(1:ntrain,:); 
ytrain = y(1:ntrain);
m = median(xtrain); 
m1 = repmat(m,ntrain,1); 
xqtrain = ones(ntrain,d); 
xqtrain(xtrain>m1) = 2;
eta0 = zeros(2,d); 
eta1 = zeros(2,d); 
for i=1:d
eta0(1,i) = sum(xqtrain(ytrain==0,i)==1)/sum(ytrain==0); 
eta0(2,i) = sum(xqtrain(ytrain==0,i)==2)/sum(ytrain==0); 
eta1(1,i) = sum(xqtrain(ytrain==1,i)==1)/sum(ytrain==1); 
eta1(2,i) = sum(xqtrain(ytrain==1,i)==2)/sum(ytrain==1); 
end
pi0 = sum(ytrain == 0)/ntrain; 
pi1 = sum(ytrain == 1)/ntrain;
ntest = n - ntrain; 
xtest = x(ntrain+1:n,:); 
ytest = y(ntrain+1:n); 
xqtest = ones(ntest,d); 
m2 = repmat(m,ntest,1); 
xqtest(xtest>m2) = 2;
yNB = zeros(ntest,1); 
for i = 1:ntest 
    tmp1 = pi0; 
    tmp2 = pi1; 
    for j = 1:d
        tmp1 = tmp1*eta0(xqtest(i,j),j); 
        tmp2 = tmp2*eta1(xqtest(i,j),j); 
    end
    if tmp1 <tmp2 
        yNB(i) = 1; 
    end
end
NBtesterror = sum(abs(yNB - ytest))/ntest; 
fprintf('The test error based on Naive Bayes classifier is %3.2f%%\n', ... 
    NBtesterror*100)