function data = load_data(fname)
    
    %read in data
    x = readtable(fname);
    
    %identify subjects
    subs = unique(x.subject);
    
    for s = 1:length(subs)
        data(s).sub = subs(s);
        rows = x.subject == subs(s);
        vars = {'subj_choice'};
        data(s).c = table2array(x(rows, vars));
        vars = {'feedback'};
        data(s).r = table2array(x(rows, vars));
        vars = {'condition'};
        data(s).cond = table2array(x(rows, vars));
        vars = {'block_num'};
        data(s).block = table2array(x(rows, vars));
        vars = {'latent_guess'};
        data(s).latent_guess = table2array(x(rows, vars));
        vars = {'age'};
        temp = table2array(x(rows, vars));
        data(s).age = temp(1);
        vars = {'age_group'};
        temp = table2array(x(rows, vars));
        data(s).ageGroup = temp(1);
        vars = {'version'};
        temp = table2array(x(rows, vars));
        data(s).version = temp(1);
        data(s).N = length(data(s).c);
        vars ={'mine_prob_win_left'};
        vars2 ={'mine_prob_win_right'};
        winprob = [table2array(x(rows, vars)), table2array(x(rows, vars2))];
        for n=1:data(s).N
            if data(s).c(n) == 0
                data(s).c(n) = 2;
            end
            [~,k] = max(winprob(n,:));
            if data(s).c(n) == k
                data(s).acc(n) = 1;
            else
                data(s).acc(n) = 0;
            end
        end
        
%compute overall subject accuracy
        data(s).overallacc = mean(data(s).acc);

    end
    
    ix = [data.overallacc]>0.6; %filter out subs who sucked
    data = data(ix); %save data without bad subs
