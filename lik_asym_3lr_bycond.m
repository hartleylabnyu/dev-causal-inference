function lik = lik_asym_3lr_bycond(x,data)
    
    % Likelihood function for Q-learning on two-armed bandit with separate
    % learning rates for each conditions
    %
    % USAGE: lik = lik_asym_3lr_bycond(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - inverse temperature
    %       x(2:4) - learning rates
    %       x(5) - stickiness
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
    lr_benev = x(2);  % learning rate, millionaire condition
    lr_adv = x(3);  % learning rate, robber condition
    lr_rand = x(4);  % learning rate, sheriff condition
    sticky = x(5);      % stickiness
    
    % initialization
    lik = 0;             % log-likelihood
    
    for n = 1:data.N
        
        if n==1 || data.block(n)~=data.block(n-1)
            v = zeros(1,2)+0.5;  % initial values
            u = zeros(1,2);
            
            
            switch data.cond(n)
                case 1 %adversarial 
                    lr = lr_adv; 
                case 2 %benevolent
                    lr = lr_benev;
                case 3 %random
                    lr = lr_rand;
            end
            
            
        end
        
        c = data.c(n);
        r = data.r(n);
        q = b*v + sticky*u;
        u = zeros(1,2); u(c) = 1;
        lik = lik + q(c) - logsumexp(q,2);
        rpe = r-v(c);
        v(c) = v(c) + lr*rpe;
        
    end
end