function lik = lik_asym_sticky_1lr(x,data)
    
    % Likelihood function for Q-learning on two-armed bandit with 
    % a single learning rate.
    %
    % USAGE: lik = lik_asym_sticky_1lr(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - inverse temperature
    %       x(2) - learning rate
    %       x(3) - stickiness
    %   data - structure with the following fields
    %          .c - [N x 1] choices
    %          .r - [N x 1] rewards
    %
    % OUTPUTS:
    %   lik - log-likelihood
    %
    % Adapted from Sam Gershman, June 2017
    
    % parameters
    b = x(1);           % inverse temperature
    lr = x(2);  % learning rate 
    sticky = x(3);      % stickiness
    
    % initialization
    lik = 0;             % log-likelihood
    
    for n = 1:data.N
        
        if n==1 || data.block(n)~=data.block(n-1) %reset values at beginning of each block
            v = zeros(1,2)+0.5;  % initial values
            u = zeros(1,2); 
        end
        
        c = data.c(n);
        r = data.r(n);
        q = b*v + sticky*u;
        u = zeros(1,2); u(c) = 1;
        lik = lik + q(c) - logsumexp(q,2);
        rpe = r-v(c);
        v(c) = v(c) + lr *rpe;
        end
        
    end
