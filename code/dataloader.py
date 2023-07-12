#%%
from datasets import load_dataset

reddit = load_dataset('reddit')
print(reddit)

reddit= reddit['train']

#%% check occurence of subreddits in dataset
from collections import Counter
subreddits = reddit['subreddit']

category_counts = Counter(subreddits)
print(category_counts.most_common())
#'AskMen', 14538
#'AskWomen', 9278

#%% filter dataset based on subreddit categories AskMen and AskWomen

askmen = reddit.filter(lambda example: example['subreddit'] == 'AskMen')
askwomen = reddit.filter(lambda example: example['subreddit'] == 'AskWomen')

#%% download as csv
askmen.to_csv("../data/askmen.csv", index=None)
askwomen.to_csv("../data/askwomen.csv", index=None)
