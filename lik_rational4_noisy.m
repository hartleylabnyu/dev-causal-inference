function [lik,latents] = lik_rational4_noisy(x,data)
    
    % Likelihood function for Q-learning on two-armed bandit with separate
    % learning rates for positive and negative prediction errors.
    % Rational/Bayesian model with added noise parameter, epsilon.
    %
    % USAGE: lik = lik_rational4_noisy(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - inverse temperature
    %       x(2) - stickiness
    %       x(3) - epsilon
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
    e = x(3);           % epsilon - intervention variability
    pz = mean(data.latent_guess);
    
    % initialization
    alpha = 1;
    beta = 1;
    lik = 0;            % log-likelihood
    
    for n = 1:data.N
        % initial values
        if n==1 || data.block(n)~=data.block(n-1)
            N = [alpha+beta alpha+beta];
            v = zeros(1,2)+(alpha/(alpha+beta));   %theta aka value of choices
            u = zeros(1,2);
        end
        
        c = data.c(n);
        r = data.r(n);
        q = b*v + sticky*u; %temp*value plus stickiness*what your last choice was
        u = zeros(1,2); u(c) = 1; %previous trial choice, 0 or 1
        lik = lik + q(c) - logsumexp(q,2); %likelihood equation
        rpe = r-v(c); %reward prediction error (received reward - previous value of choice)
        
        if r == 1 %if reward received = 1
            if data.cond(n)==1 %if condition = adversarial
                psi = v(c)*(1-pz)/(e.*pz+v(c)*(1-pz));
            elseif data.cond(n)==2 %if condition = benevolent
                psi = v(c)*(1-pz)/((1-e).*pz+v(c)*(1-pz));
            elseif data.cond(n)==3 %if condition = random
                psi = v(c)*(1-pz)/(pz/2 + v(c)*(1-pz));
            end
        else %if reward = 0
            if data.cond(n)==1 %and condition = adversarial
                psi = (1-v(c))*(1-pz)/((1-e).*pz + (1-v(c))*(1-pz)); %then this is probability of agent not intervening
            elseif data.cond(n)==2 %and condition = benevolent
                psi = (1-v(c))*(1-pz)/(e.*pz + (1-v(c))*(1-pz)); %then this is probability of agent not intervening
            elseif data.cond(n)==3 %and condition = random
                psi = (1-v(c))*(1-pz)/(pz/2 + (1-v(c))*(1-pz));
            end
        end
        
        lr = 1/N(c); %learning rate = 1/(N + alpha + beta); N is your belief that agent did not intervene, summed over all past trials
        v(c) = v(c) + lr*rpe*psi; %full update rule (theta hat = theta + learning rate * PE * psi)
        N(c) = N(c) + psi;
        
        if nargout > 1
            latents.choiceprob(n,:) = exp(q)./sum(exp(q)); %softmax
            latents.lr(n,1) = lr*psi; %learning rates
            latents.latent_guess(n,1) = 1-psi; %belief probability of agent intervening
        end
    end