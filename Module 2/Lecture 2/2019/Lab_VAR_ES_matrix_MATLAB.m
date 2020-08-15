
% 
% LAB: VAR and ES portfolio
% Thierry Roncalli, Introduction to Risk Parity and Budgeting, CRC Press, 2014
% Example page 75
%
% Fitch Learning UK
% Jan 2018
%

close all 
clear all

% STEP 1: inputs
% three stocks A,B,C
p=[244 135 315]';
x=[0.5203 0.1439 0.3358]';
mu=[50 30 20]'./10000;
vol=[2 3 1]./100;
rho=[1 0.5 0.25;0.5 1 0.6;0.25 0.6 1];

Sigma=[4.0000e-004  3.0000e-004  5.0000e-005;
  3.0000e-004  9.0000e-004  1.8000e-004;
  5.0000e-005  1.8000e-004  1.0000e-004]
  
alpha=0.99;


% STEP 2: calculations

% compute VAR
VAR_x = -x'*mu + norminv(alpha)*sqrt(x'*Sigma*x);

% compute ES
ES_x = -x'*mu + (sqrt(x'*Sigma*x))/(1-alpha)*normpdf(norminv(alpha));


% STEP 3: output
VAR_x
ES_x


