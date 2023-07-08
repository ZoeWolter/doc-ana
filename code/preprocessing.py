#%% import relevant packages -----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re 
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet') 
from nltk.stem import WordNetLemmatizer 
import random 
from wordcloud import WordCloud 
from PIL import Image
from bertopic import BERTopic
import csv
from bertopic.representation import KeyBERTInspired

#%% load data from csv files -----------------------------------------------------------

askmen = pd.read_csv('../data/askmen.csv') 
askwomen = pd.read_csv('../data/askwomen.csv') 

content_men = np.array(askmen['content']) 
content_women = np.array(askwomen['content']) 

#%% data inspection -----------------------------------------------------------------

posts_men = len(content_men)
posts_women = len(content_women)

print(f'total amount of posts in AskMen: {posts_men}')
print(f'total amount of posts in AskWomen: {posts_women}')

#check length of posts (#tokens) 
post_lengths_men = [len(post.split()) for post in content_men] 
post_lengths_women = [len(post.split()) for post in content_women] 

fig, ax = plt.subplots(1,2)
fig.suptitle('Distribution of post lengths')

sns.countplot(x = post_lengths_men, color = 'blue', ax = ax[0])
ax[0].set_title('AskMen\naverage: 257')
ax[0].set_xlabel('Post length')
ax[0].set_ylabel('Frequency')

sns.countplot(x = post_lengths_women, color = 'red', ax = ax[1])
ax[1].set_title('AskWomen\naverage: 250')
ax[1].set_xlabel('Post length')
ax[1].set_ylabel('Frequency')

fig.show()

print(f'Average post length AskMen: {np.round(np.mean(post_lengths_men))}')
print(f'Average post length AskWomen: {np.round(np.mean(post_lengths_women))}')

#%% data preprocessing -----------------------------------------------------------------

def preprocess(posts):
    '''
    Function to preprocess reddit posts

    Parameters:
    posts (np.array) : reddit posts

    Returns:
    posts : preprocessed reddit posts
    '''

    lemmatizer = WordNetLemmatizer() 
    stopwords = list(np.loadtxt('../data/eng_stop_words.txt', dtype='str')) 
    
    for i in range(len(posts)):
        posts[i] = posts[i].lower() 
        posts[i] = re.sub("'", "", posts[i]) # account for negation short forms (for instance: can't -> cant)
        posts[i] = posts[i].replace("\n", "")  # remove line breaks
        posts[i] = re.sub(r"[^a-zA-Z0-9]", " ", posts[i]) # remove special characters
        # remove stop words and lemmatize words
        words = posts[i].split()
        words_new = [lemmatizer.lemmatize(word) for word in words if word not in stopwords and len(word) > 1] 
        posts[i] = " ".join(words_new)
    return posts

# use function to preprocess reddit posts of AskMen and AskWomen subbredits
content_men = preprocess(content_men)
content_women = preprocess(content_women)

#%% generate wordclouds ----------------------------------------------------------------

def make_wordcloud(x, imagename, maxfont, maxwords, backcolor, textcolorfunc):
    '''
    Function to create a wordcloud 
    '''

    words = []
    for i in range(len(x)):
        words.append(x[i])   
    words = ','.join(words)
    icon  = np.array(Image.open(imagename)) 
    frequent_words= WordCloud(max_font_size=maxfont, max_words=maxwords, background_color=backcolor, mask=icon, random_state=42).generate(words)
    return frequent_words.recolor(color_func=textcolorfunc, random_state=3) 

# define text color functions
def blue_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(230, 100%%, %d%%)" % random.randint(20, 80) 
def red_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(0, 100%%, %d%%)" % random.randint(20, 80)

frequent_words_men = make_wordcloud(content_men, '../img/men.png', 30, 200,'white', blue_color_func)
frequent_words_women = make_wordcloud(content_women, '../img/women.png', 30, 200, 'white', red_color_func)

f = plt.figure(figsize=(10,10))
f.add_subplot(1,1,1)
plt.imshow(frequent_words_men, interpolation='bilinear')
plt.axis('off')
plt.savefig('../img/AskMen_WordCloud.png')
plt.show()

f = plt.figure(figsize=(10,10))
f.add_subplot(1,1,1)
plt.imshow(frequent_words_women, interpolation='bilinear')
plt.axis('off')
plt.savefig('../img/AskWomen_WordCloud.png')
plt.show()


#%% Topic Analysis using BERTopic: 50 TOPICS -------------------------------------------
# not required to run as data is already stored in folder '/data'!

# create combined data set (men & women)
content_combined = np.concatenate((content_men, content_women))

# define model and reduce topics to 50 for further analysis (nr_toptics = 51, because topic -1 will be ignored later)
topic_model_combined = BERTopic(representation_model=KeyBERTInspired(), calculate_probabilities=True, nr_topics=51) # using representation model (KeyBERT) to get more pronounced topics (without unnecessary verbs etc.) 
reduced_topics, probs = topic_model_combined.fit_transform(content_combined)
topic_info_combined = pd.DataFrame(topic_model_combined.get_topic_info()) # get topic info as dataframe

# get more detailed info about each topic
topic_details_combined = []
for i in range(0, len(topic_info_combined)):
    topic_details_combined.append(topic_model_combined.get_topic(i))

# get detailed info about each document (each reddit post)
document_info_combined = pd.DataFrame(topic_model_combined.get_document_info(content_combined))

# get embeddings of each topic
embeddings = topic_model_combined.topic_embeddings_

# save all outputs as csv
outfile = open('../data/BerTopic_50/topic_details_combined.csv','w')
writer = csv.writer(outfile)
writer.writerows([[topic] for topic in topic_details_combined])
outfile.close()

emb = pd.DataFrame(embeddings)
emb.to_csv('../data/BerTopic_50/embeddings.csv', index=False, header=False)  

topic_info_combined.to_csv('../data/BerTopic_50/topic_info_combined.csv')
document_info_combined.to_csv('../data/BerTopic_50/document_info_combined.csv')

#%% Visualizations for topics ----------------------------------------------------------

fig_combined_topics = topic_model_combined.visualize_topics()
fig_combined_topics.write_html('../img/fig_combined_topics.html')

fig_combined_documents = topic_model_combined.visualize_documents(content_combined)
fig_combined_documents.write_html('../img/fig_combined_documents.html')

fig_combined_barchart = topic_model_combined.visualize_barchart(top_n_topics=20)
fig_combined_barchart.write_html('../img/fig_combined_barchart.html')

fig_combined_hierarchy = topic_model_combined.visualize_hierarchy()
fig_combined_hierarchy.write_html('../img/fig_combined_hierarchy.html')

fig_combined_heatmap = topic_model_combined.visualize_heatmap()
fig_combined_heatmap.write_html('../img/fig_combined_heatmap.html')

fig_combined_distribution = topic_model_combined.visualize_distribution(probs[0])
fig_combined_distribution.write_html('../img/fig_combined_distribution.html')


#%% Create single csv file with all topic infos -----------------------------------

# document info
doc_info = pd.read_csv('../data/BerTopic_50/document_info_combined.csv')

# add variable for men and women
doc_info['gender'] = np.nan
doc_info.loc[:posts_men,'gender'] = 'men'  
doc_info.loc[posts_men:,'gender'] = 'women'

# get count per gender
gender_counts = doc_info.groupby(['Name', 'gender'])['Name'].size().reset_index(name='count')
gender_counts  = gender_counts.pivot(index='Name', columns='gender', values='count').reset_index()

# add topic info
topic_info = pd.read_csv('../data/BerTopic_50/topic_info_combined.csv', usecols=['Name','Topic','Count','Representation','Representative_Docs'])
df = pd.merge(gender_counts, topic_info , on='Name', how='outer')

df.rename(columns={'men': 'Count_men', 'women': 'Count_women', 'Topic': 'Topic_id'}, inplace=True) #rename columns
print('AskMen posts in topic -1:', df[df['Topic_id']==-1]['Count_men'][0])
print('AskWomen posts in topic -1:', df[df['Topic_id']==-1]['Count_women'][0])
df = df[df['Topic_id'] != -1] #remove topic -1 
df = df.sort_values('Topic_id') #reorder
df = df.reset_index(drop=True) #reset index
df['Name'] = df['Name'].str.replace(r'\d+_', '') #remove prefix topic id in topic names

#%% add embeddings

#load embeddings
embeddings = pd.read_csv('../data/BerTopic_50/embeddings.csv', header = None)
embeddings = embeddings.iloc[1:] #delete embedding for topic -1

#%% alternative: calculate topic embeddings instead of using BerTopic embeddings
'''
#from gensim.models import Word2Vec
import gensim.downloader

topics = list(df['Name'])
#use pretrained Word2Vec (trained on twitter data) -> glove embeddings 
glove_vectors = gensim.downloader.load('glove-twitter-200') #takes several minutes to load

embeddings = []
for topic in topics:
    topic_words = topic.split("_")[1:]  
    word_embeddings = [glove_vectors[word] for word in topic_words if word in glove_vectors]  
    if word_embeddings:
        topic_embedding = np.mean(word_embeddings, axis=0)  # use mean of word embeddings as whole topic embedding
    else:
        topic_embedding = None
    embeddings.append(topic_embedding)
'''
#%%use PCA to reduce embeddings to 2 dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
coordinates_2d = pca.fit_transform(embeddings)

#%% alternative dimensionality reduction techniques
'''
from sklearn.manifold import TSNE
e = np.array(embeddings)
tsne = TSNE(n_components=2)  
coordinates_2d = tsne.fit_transform(e)
#%%
from sklearn.manifold import MDS
mds= MDS(n_components=2)
coordinates_2d = mds.fit_transform(embeddings)
#%%
import umap
reducer = umap.UMAP()
coordinates_2d = reducer.fit_transform(embeddings)
'''
#%%
df_coordinates = pd.DataFrame(coordinates_2d, columns=['x', 'y'])
data_added = pd.concat([df, df_coordinates], axis=1)

#export 
data_added.to_csv('../data/topics.csv', sep=';', index=False)

#%% share of genders per top10 topics ---------------------------------------------

# load data from csv
topics = pd.read_csv('../data/topics.csv', sep = ';') 

# add column with shares from absolute values
topics['share_men'] = topics['Count_men'] / topics['Count']
topics['share_women'] = topics['Count_women'] / topics['Count']

# just relevant topics and columns
top10 = topics.sort_values(by=['Count'], ascending = False).loc[:9, ['Name', 'share_men', 'share_women']]

# plot
ax = top10.plot.bar(stacked = True, color = ['blue', 'red'])
plt.tight_layout()
ax.set_xticklabels(top10['Name'], rotation = 45, ha = 'right')
ax.legend(['AskMen', 'AskWomen'])
ax.set_title('Share of AskMen and AskWomen per topic (unweighted)')
plt.axhline(y = 0.5, color = 'black', linestyle = 'dashed')
plt.savefig('../img/top10_shares.svg', bbox_inches = 'tight')
plt.show()

# add column with adjusted shares from absolute values
topics['share_men_adjusted'] =(topics['Count_men'] / 14538) / (topics['Count_men'] / 14538 + topics['Count_women'] / 9278)
topics['share_women_adjusted'] =(topics['Count_women'] / 9278) / (topics['Count_men'] / 14538 + topics['Count_women'] / 9278)

# just relevant topics and columns
top10 = topics.sort_values(by=['Count'], ascending = False).loc[:9, ['Name', 'share_men_adjusted', 'share_women_adjusted']]

# plot
ax = top10.plot.bar(stacked = True, color = ['blue', 'red'])
plt.tight_layout()
ax.set_xticklabels(top10['Name'], rotation = 45, ha = 'right')
ax.legend(['AskMen', 'AskWomen'])
ax.set_title('Share of AskMen and AskWomen per topic (weighted)')
plt.axhline(y = 0.5, color = 'black', linestyle = 'dashed')
plt.savefig('../img/top10_shares_adjusted.svg', bbox_inches = 'tight')
plt.show()
# %%
