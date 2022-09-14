%% Value Function Iteration for Non-Stochastic Growth Model
% Created by Stefano Lord Medrano using Kuan's notes
% Last modified: 09/10/2022
% University of Wisconsin-Madison
% Econ 899: Computational Methods
% Problem Set 1

% Clean workspace
clear all;
close all;
clc;
tic

%% Parameters and grid
% Parameters
beta = 0.99;      % Discount factor
alpha = 0.36;     % Capital share
delta = 0.025;    % Depreciation rate
tol = 1e-3;       % Tolerance level

% Construct capital grid
k_lb = 0.01; % Lower bound of grid points
k_ub = 75;    % Upper bound of grid points
n = 1000;     % Size of the asset grid
inc=(k_ub-k_lb)/(n-1); % Increments
k=(k_lb:inc:k_ub)';    % Capital grid
kp=repmat(k,1,n);    % Create a grid for next period asset choice

% Initial guess of the value functions and decision rules
v0=zeros(n,1);
v1=zeros(n,1);
optindex=ones(n,1);

% Counter of iterations
iter=0;

% Initialize supnorm of values function in two consecutive iterations
supnorm=1;

%% Value function iteration
tic;
%parpool(4); % start the parallel pool in matlab. The number in parenthesis is the number of cores in CPU
while supnorm>=tol  
    for i=1:n % sequential loop
        %parfor i=1:n % parallel loop
        kp_temp=kp(:,i);
        c=k(i).^alpha+(1-delta)*k(i)-kp_temp;
        c=(c>0).*c+(c<=0).*0;
        [v1(i),optindex(i)]=max(log(c)+beta*v0);
    end
supnorm=norm(v0-v1); % update supnorm
v0=v1;
iter=iter+1;
fprintf('Iteration # %2d \tSupnorm: %g \n',iter,supnorm);
end 
toc;

%% Value Function Iteration for Stochastic Growth Model
% Last modified: 09/10/2022
% University of Wisconsin-Madison
% Econ 899: Computational Methods
% Problem Set 1

% Clean workspace
clear all;
close all;
clc;
tic

%% Parameters and grid
% Parameters
beta = 0.99;      % Discount factor
alpha = 0.36;     % Capital share
delta = 0.025;    % Depreciation rate
tol = 1e-3;       % Tolerance level
amat = [1.25 0.2]'; % Possible values of TFP
prob = [0.977 0.023 ; 0.074 0.926]; % Transition matrix

% Construct capital grid
k_lb = 0.01; % Lower bound of grid points
k_ub = 75;    % Upper bound of grid points
n = 1000;     % Size of the asset grid
inc=(k_ub-k_lb)/(n-1); % Increments
k=(k_lb:inc:k_ub)';    % Capital grid
kp=repmat(k,1,n);    % Create a grid for next period asset choice

% Initial guess of the value functions and decision rules
v0=zeros(n,2);
v1=zeros(n,2);
optindex=ones(n,2);

% Counter of iterations
iter=0;

% Initialize supnorm of values function in two consecutive iterations
supnorm=1;

%% Value function iteration
tic;
while supnorm>=tol  
    parfor j=1:2
    %for j=1:2
    for i=1:n % sequential loop
        kp_temp=kp(:,i);
        c=amat(j,1)*k(i,1).^alpha+(1-delta)*k(i,1)-kp_temp;
        c=(c>0).*c+(c<=0).*0;
        [v1(i,j),optindex(i,j)]=max(log(c)+beta*v0*prob(j,:)');
    end
    end
supnorm=norm(v0-v1); % update supnorm
v0=v1;
iter=iter+1;
fprintf('Iteration # %2d \tSupnorm: %g \n',iter,supnorm);
end 
toc;

% Extract policy functions
cap = k(optindex);

% Savings (k'-k)
sav = cap-k;
%% Plot
figure
plot(k,v0,'Linewidth',1)
xlabel('$k$','Interpreter','latex','FontSize',14)
ylabel('$V(k)$','Interpreter','latex','FontSize',14)
legend('Good State','Bad State','Location','Northwest','Interpreter','latex','FontSize',10);

figure
plot(k,cap,'Linewidth',1)
xlabel('$k$','Interpreter','latex','FontSize',14)
ylabel('$k^{^{,}}(k)$','Interpreter','latex','FontSize',14)
legend('Good State','Bad State','Location','Northwest','Interpreter','latex','FontSize',10);

figure
plot(k,sav,'Linewidth',1)
xlabel('$k$','Interpreter','latex','FontSize',14)
ylabel('$k^{^{,}}(k)-k$','Interpreter','latex','FontSize',14)
legend('Good State','Bad State','Location','Northeast','Interpreter','latex','FontSize',10);

