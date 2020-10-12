function [lik,latents] = lik_rational_adaptive(x,data)
    
    % Likelihood function for Q-learning on two-armed bandit with separate
    % learning rates for positive and negative prediction errors and
    % unknown intervention probability.
    %
    % USAGE: lik = lik_rational_adaptive(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - inverse temperature
    %       x(2) - stickiness
    %   data - structure with the following fields
    %          .c - [N x 1] choices
    %          .r - [N x 1] rewards
    %
    % OUTPUTS:
    %   lik - log-likelihood
    %
    % Sam Gershman, June 2017
    
    % parameters
    b = x(1);           % inverse temperature
    sticky = x(2);      % stickiness
    
    % initialization
    alpha = 1;
    beta = 1;
    alpha_g = 1;
    beta_g = 1;
    pz = alpha_g/(alpha_g + beta_g);    % initial intervention probability
    lik = 0;            % log-likelihood
    
    for n = 1:data.N
        
        if n==1 || data.block(n)~=data.block(n-1)
            N = [alpha+beta alpha+beta];
            v = zeros(1,2)+(alpha/(alpha+beta));  % initial values
            u = zeros(1,2);
            t = 0;
        end
        
        t = t + 1;
        c = data.c(n);
        r = data.r(n);
        q = b*v + sticky*u;
        u = zeros(1,2); u(c) = 1;
        lik = lik + q(c) - logsumexp(q,2);
        if isnan(lik); keyboard; end
        rpe = r-v(c);
        
        if r == 1
            if data.cond(n)==1
                psi = 1;
            elseif data.cond(n)==2
                psi = v(c)*(1-pz)/(pz+v(c)*(1-pz));
            elseif data.cond(n)==3
                psi = v(c)*(1-pz)/(pz/2 + v(c)*(1-pz));
            end
        else
            if data.cond(n)==1
                psi = (1-v(c))*(1-pz)/(pz + (1-v(c))*(1-pz));
            elseif data.cond(n)==2
                psi = 1;
            elseif data.cond(n)==3
                psi = (1-v(c))*(1-pz)/(pz/2 + (1-v(c))*(1-pz));
            end
        end
        
        lr = 1/N(c);
        v(c) = v(c) + lr*rpe*psi;
        N(c) = N(c) + psi;
        
        if nargout > 1
            latents.choiceprob(n,:) = exp(q)./sum(exp(q));
            latents.lr(n,1) = lr*psi;
            latents.latent_guess(n,1) = 1-psi;
            latents.pz(n,1) = pz;
        end
        
        % update intervention probability
        lrz = 1/(n + alpha_g + beta_g); %learning rate for agent intervention probability
        pz = pz + lrz*(1-psi-pz); %belief of intervention probability
    end