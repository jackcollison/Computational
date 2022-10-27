%% Value Function Iteration for Stochastic Growth Model
% Created by Stefano Lord Medrano using Professor Eric Sims Lecture Notes
% Last modified: 09/10/2022
% University of Wisconsin-Madison
% Econ 899: Computational Methods
% Problem Set 1

% Clean workspace
clear all;
close all;
clc;
tic

% set parameters
global v0 beta delta alpha kmat k0 a0 amat prob j
 alpha = 0.36; % capital's share
 beta = 0.99;  % discount factor
 delta = 0.025; % depreciation rate 
 amat = [1.25 0.2]'; % possible values of TFP
 prob = [0.977 0.023 ; 0.074 0.926]; % transition matrix

% Tolerance criterion and number of iterations
tol = 0.01;
maxits = 1000;
dif = tol+1000;
its=0;

%% Grid for capital
% Deterministic steady-state for initial guess
kstar = (alpha/(1/beta - (1-delta)))^(1/(1-alpha)); % steady state k
cstar = kstar^(alpha) - delta*kstar;
istar = delta*kstar;
ystar = kstar^(alpha);

% Grid for capital and matrix for initial guess for value function
 kgrid = 99; % grid points + 1
 kmin = 0.25*kstar;
 kmax = 1.75*kstar;
 grid = (kmax-kmin)/kgrid;
 kmat = kmin:grid:kmax;
 kmat = kmat';
 [N,n] = size(kmat);
 v0 = zeros(N,2);

% Alternative grid
% kgrid = 99; % grid points + 1
% kmin = 0.01;
% kmax = 75;
% grid = (kmax-kmin)/kgrid;
% kmat = kmin:grid:kmax;
% kmat = kmat';
% [N,n] = size(kmat);
% v0 = zeros(N,2);

%% Iteration of value function
while dif>tol & its < maxits 
 for j=1:2      % Outside loop for states of nature
    for i=1:N   % Inside loop for iteration of value function
    k0 = kmat(i,1);
    a0 = amat(j,1);
    k1 = fminbnd(@valfuna1_stoch,kmin,kmax); 
    v1(i,j) = -valfuna1_stoch(k1);
    k11(i,j) = k1;
    end
 end
    dif= norm(v1-v0); 
    v0=v1;
    its = its+1
end
toc
%% Policy functions
for j=1:2
for i=1:N
 con(i,j) = amat(j,1)*kmat(i,1)^(alpha) - k11(i,j) + (1-delta)*kmat(i,1);
 inv(i,j) = k11(i,j) - (1-delta)*kmat(i,1);
 y(i,j) = con(i,j) + k11(i,j) - (1-delta)*kmat(i,1);
end
end

% Parallelization
% parfor j=1:3
% for i=1:N
%  con(i,j) = amat(j,1)*kmat(i,1)^(alpha) - k11(i,j) + (1-delta)*kmat(i,1);
%  inv(i,j) = k11(i,j) - (1-delta)*kmat(i,1);
%  y(i,j) = con(i,j) + k11(i,j) - (1-delta)*kmat(i,1);
% end
% end


%% Plots
figure
plot(kmat,v1,'Linewidth',1)
xlabel('$k$','Interpreter','latex','FontSize',14)
ylabel('$V(k)$','Interpreter','latex','FontSize',14)
legend('Good State','Bad State','Location','Northwest','Interpreter','latex','FontSize',10);

figure
plot(diff(v1),'Linewidth',1)
%xlabel('$k$','Interpreter','latex','FontSize',14)
ylabel('$\Delta V(k)$','Interpreter','latex','FontSize',14)
legend('Good State','Bad State','Location','Northwest','Interpreter','latex','FontSize',10);

figure
plot(kmat,k11,'Linewidth',1)
xlabel('$k$','Interpreter','latex','FontSize',14)
ylabel('$k^{^{,}}(k)$','Interpreter','latex','FontSize',14)
legend('Good State','Bad State','Location','Northwest','Interpreter','latex','FontSize',10);

figure
plot(diff(k11),'Linewidth',1)
%xlabel('$k$','Interpreter','latex','FontSize',14)
ylabel('$\Delta k^{^{,}}(k)$','Interpreter','latex','FontSize',14)
legend('Good State','Bad State','Location','Northwest','Interpreter','latex','FontSize',10);

figure
plot(kmat,con,'Linewidth',1)
xlabel('$k$','Interpreter','latex','FontSize',14)
ylabel('$c(k)$','Interpreter','latex','FontSize',14)
legend('Good State','Bad State','Location','Northwest','Interpreter','latex','FontSize',10);

figure
plot(kmat,inv,'Linewidth',1)
xlabel('$k$','Interpreter','latex','FontSize',14)
ylabel('$x(k)$','Interpreter','latex','FontSize',14)
legend('Good State','Bad State','Location','Northwest','Interpreter','latex','FontSize',10);

figure
plot(kmat,y,'Linewidth',1)
xlabel('$k$','Interpreter','latex','FontSize',14)
ylabel('$y(k)$','Interpreter','latex','FontSize',14)
legend('Good State','Bad State','Location','Northwest','Interpreter','latex','FontSize',10);