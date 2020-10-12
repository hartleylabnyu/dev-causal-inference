function [results, bms_results] = fit_models(data, models)

%[results, bms_results] = fit_models(experiment,results,models)
    
    % Fit RL models using MFIT.
    %
    % USAGE: [results, bms_results] = fit_models(data)
    
    % USAGE (FOR REAL): [results, bms_results] = fit_models(exp num, data)
    % Copy/paste the line above into the command window. Include exp num as
    % the actual number of the experiment (1 or 2) - this will change the
    % path to your data. Leave the word 'data'
    %
    % INPUTS:
    %   data - [S x 1] structure array of data for S subjects
    %
    % OUTPUTS:
    %   results - [M x 1] model fits
    %   bms_results - Bayesian model selection results
    %
    % Sam Gershman, Jun 2016
    
    %specify the likelihood functions
    likfuns = {'lik_asym_sticky_1lr','lik_asym_sticky_2lr','lik_asym_3lr_bycond', 'lik_rational_pzedit', 'lik_rational_adaptive', 'lik_rational4_noisy', 'lik_rational4'}; %full model set
    
    if nargin < 2; models = 1:length(likfuns); end
    
    for mi = 1:length(models)
        m = models(mi);
        disp(['... fitting model ',num2str(m)]);
        
        switch likfuns{m}
            
          case 'lik_asym_sticky_1lr'
                param(1) = struct('name','invtemp','logpdf',@(x) log(gampdf(x,4.82,0.88)),'lb',1e-3,'ub',20);
                param(2) = struct('name','lr','logpdf',@(x) 0,'lb',0,'ub',1);
                param(3) = struct('name','sticky','logpdf',@(x) 0,'lb',-5,'ub',5);
          
          case 'lik_asym_sticky_2lr'
                param(1) = struct('name','invtemp','logpdf',@(x) log(gampdf(x,4.82,0.88)),'lb',1e-3,'ub',20);
                param(2) = struct('name','lr_pos','logpdf',@(x) 0,'lb',0,'ub',1);
                param(3) = struct('name','lr_neg','logpdf',@(x) 0,'lb',0,'ub',1);
                param(4) = struct('name','sticky','logpdf',@(x) 0,'lb',-5,'ub',5);
             
          case 'lik_asym_3lr_bycond'
                param(1) = struct('name','invtemp','logpdf',@(x) log(gampdf(x,4.82,0.88)),'lb',1e-3,'ub',20);
                param(2) = struct('name','lr_benev','logpdf',@(x) 0,'lb',0,'ub',1);
                param(3) = struct('name','lr_adv','logpdf',@(x) 0,'lb',0,'ub',1);
                param(4) = struct('name','lr_rand','logpdf',@(x) 0,'lb',0,'ub',1);
                param(5) = struct('name','sticky','logpdf',@(x) 0,'lb',-5,'ub',5);
                
          case 'lik_rational4'
                param(1) = struct('name','invtemp','logpdf',@(x) log(gampdf(x,4.82,0.88)),'lb',1e-3,'ub',20);
                param(2) = struct('name','sticky','logpdf',@(x) 0,'lb',-5,'ub',5);
                
          case 'lik_rational4_noisy'
                param(1) = struct('name','invtemp','logpdf',@(x) log(gampdf(x,4.82,0.88)),'lb',1e-3,'ub',20);
                param(2) = struct('name','sticky','logpdf',@(x) 0,'lb',-5,'ub',5);
                param(3) = struct('name','epsilon','logpdf',@(x) 0,'lb',0,'ub',0.5);
                
          case 'lik_rational_pzedit'
                param(1) = struct('name','invtemp','logpdf',@(x) log(gampdf(x,4.82,0.88)),'lb',1e-3,'ub',20);
                param(2) = struct('name','sticky','logpdf',@(x) 0,'lb',-5,'ub',5);
                
          case 'lik_rational_adaptive'
                param(1) = struct('name','invtemp','logpdf',@(x) log(gampdf(x,4.82,0.88)),'lb',1e-3,'ub',20);
                param(2) = struct('name','sticky','logpdf',@(x) 0,'lb',-5,'ub',5);
                likfuns{m} = 'lik_rational_adaptive';
         
        end
        
        fun = str2func(likfuns{m});
        results(m) = mfit_optimize(fun,param,data);
        clear param
    end
    
    % Bayesian model selection
    if nargout > 1
        bms_results = mfit_bms(results,1);  % use BIC, because asym_sticky Hessian seems to be degenerate
    end