############################
### DESCRIPTIVE ANALYSIS ###
############################

#%% import relevant packages -----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% load data from csv files -----------------------------------------------------------

askmen = pd.read_csv('../data/askmen.csv') 
askwomen = pd.read_csv('../data/askwomen.csv') 
content_men = np.array(askmen['content']) 
content_women = np.array(askwomen['content']) 


share_50 = pd.read_csv('../data/shares/topics_share_gender.csv')



#%% share of genders per topic top10

#ToDo -> ändern

top10 = share_50.loc[2:23, ['Name', 'gender', 'share']].set_index('Name')
top10 = pd.pivot(top10, columns = 'gender', values = 'share')

# plot
ax = top10.plot.bar(stacked = True)
plt.tight_layout()
plt.savefig('../img/top10_shares_50.png')

# %%
