# The rational use of causal inference to guide reinforcement learning strengthens with age

Data and analysis code for: Cohen, A.O.\*, Nussenbaum, K\*, Dorfman, H.M., Gershman, S.J., & Hartley, C.A. (2020). [The rational use of causal inference to guide reinforcement learning strengthens with age.](https://www.nature.com/articles/s41539-020-00075-3) *npj Science of Learning.*

Anonymized trial-wise data for all participants are provided in anonymized_mining_data.csv. A key for the variable names in the header of this csv file is provided below. The data were minimally processed for flexible use in behavioral analyses and model-fitting. Both the Rmarkdown code and the Matlab code take this file as input. Summarized data from reported simulations can be found in accTable_realParams_sims.txt and are used in the Rmarkdown file to generate figures. Model-fitting analyses require the [mfit package](https://github.com/sjgershm/mfit). Data were loaded into Matlab using load_data.m and separated by age groups using sepDataAgeGroup.m prior to model-fitting.

---
#### Variable Key for anonymized_mining_data.csv 
*subject:* randomly generated subject ID  
*usable:* subjects to be included in analyses (1 = include)  
*age:* exact age of participant  
*age_group:* participant’s age group designation  
*gender:* 0 = male, 1 = female  
*version:* the territory and trial presentation order   
*block_num:* 1 = first learning block, 2 = second learning block, 3 = third learning block  
*condition:* condition (1 = robber, 2 = millionaire, or 3 = sheriff)  
*trial_num:* trial number  
*trial_in_block:* trial number within each learning block  
*mine_prob_win_left:* probability of a positive outcome for the stimulus on the left side  
*mine_prob_win_right:* probability of a positive outcome for the stimulus on the right side  
*subj_choice:* button press (0 = right, 1 = left)  
*feedback:* reward feedback received (0 = negative outcome, 1 = positive outcome)  
*latent_guess:* button press for subject guess about latent agent intervention (0 = no, agent did not intervene, 1 = yes, agent did intervene)  
*optimal_choice:* whether the subject chose the better mine (0 = no, 1 = yes)  
*choice_RT:* choice reaction time  
