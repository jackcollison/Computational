function val=valfuna1_stoch(k)
% This program gets the value function for a neoclassical growth model with
% uncertainty and log utility

global v0 beta delta alpha kmat k0 a0 prob j

% g = interp1(kmat,v0,k,'linear'); % smooths out previous value function
g = interp1(kmat,v0,k,'spline');
c = a0*k0^alpha - k + (1-delta)*k0; % consumption
if c<=0
val = -888888888888888888-800*abs(c); % keeps it from going negative
else
val = log(c) + beta*(g*prob(j,:)');
end
val = -val; % make it negative since we're maximizing and code is to minimize.