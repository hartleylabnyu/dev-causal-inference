function lik = lik_asym_sticky_2lr(x,data)
    
    % Likelihood function for Q-learning on two-armed bandit with separate
    % learning rates for positive and negative outcomes in different
    % conditions.
    %
    % USAGE: lik = lik_asym_sticky_2lr(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - inverse temperature
    %       x(2:3) - learning rates
    %       x(4) - stickiness
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
    lr_pos = x(2);  % learning rate (positive outcome)
    lr_neg = x(3);  % learning rate (negative outcome)
    sticky = x(4);      % stickiness
    
    % initialization
    lik = 0;             % log-likelihood
    
    for n = 1:data.N
        
        if n==1 || data.block(n)~=data.block(n-1) %reset values at beginning of each block
            v = zeros(1,2)+0.5;  % initial values
            u = zeros(1,2); 
            lr = [lr_pos lr_neg];
        end
        
        c = data.c(n);
        r = data.r(n);
        q = b*v + sticky*u;
        u = zeros(1,2); u(c) = 1;
        lik = lik + q(c) - logsumexp(q,2);
        rpe = r-v(c);
        if r == 1
            v(c) = v(c) + lr(1)*rpe;
        else
            v(c) = v(c) + lr(2)*rpe;
        end
        
    end