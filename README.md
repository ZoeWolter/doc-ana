# Document Analysis | Project 
*Daniela Blumberg, Nele Hapig, Zoé Wolter*

## How to access the website

1. Clone this repository or download it as a [zip-file](https://github.com/ZoeWolter/doc-ana/archive/main.zip).
2. Download the data from the [cloud](https://cloud.uni-konstanz.de/index.php/s/nNfEsJRZd72rzqH) and store it as folder called `data/`. 
3. Start a terminal / cmd session in the folder `doc-ana/` and enter the command:

        python -m http.server

4. Then open the website on your localhost via `http://localhost:8000/`.


## How to run the code

1. Clone this repository or download it as a [zip-file](https://github.com/ZoeWolter/doc-ana/archive/main.zip).
2. Download the data from the [cloud](https://cloud.uni-konstanz.de/index.php/s/nNfEsJRZd72rzqH) and store it as folder called `data/`. 
3. Activate the virtual environment:

        source ./env/bin/activate

4. Install modules in requirements.txt:

        pip install -r requirements.txt 

5. Run the code in the folder `data/`. The file `dataloader.py` downloads the [reddit posts from Hugging Face](https://huggingface.co/datasets/webis/tldr-17) and `preprocessing.py` contains the code for preprocessing, BERTopic modelling, and other analyses. 


## Project structure

This project is structured as follows:

```
├── doc-ana 
│   ├── code --------------- folder which contains the code
│   │   ├── preprocessing.py ------- code for preprocessing and BERTopic modelling
│   ├── data --------------- folder which contains the data (can be found in cloud due to Git LFS storage limits)
│   │   ├── BERTopic_50 ------------ data for 50 topic from BERTopic with preprocessed data 
│   │   ├── BERTopic_50_raw_data --- data for 50 topic from BERTopic with unprocessed data
│   │   ├── askmen.csv ------------- AskMen subreddit data
│   │   ├── askwomen.csv ----------- AskWomen subreddit data
│   │   ├── topics_raw_data.csv ---- topics, count, embeddings etc. of unprocessed data
│   │   ├── topics.csv ------------- topics, count, embeddings etc. of preprocessed data
│   │   ├── eng_stop_words.txt ----- file with English stopwords
│   │   ├── dataloader.py --- data for 50 topic from BERTopic with preprocessed data
│   ├── img ---------------- folder with all images from analysis with preprocessed data
│   ├── img_raw_data ------- folder with all images from analysis with unprocessed data 
│   ├── d3.js -------------- d3 library
│   ├── index.css ---------- css style formatting
│   ├── index.html --------- html with website content
│   ├── index.js ----------- visualizations
│   ├── requirements.txt --- contains all modules used in the project 
```


## References

- Data from Völske, M., Potthast, M., Syed, S., & Stein, B. (2017). TL;DR: Mining Reddit to Learn Automatic Summarization. In *Proceedings of the Workshop on New Frontiers in Summarization* (pp. 59–63). Association for Computational Linguistics.
- Data can be downloaded [here from Hugging Face](https://huggingface.co/datasets/webis/tldr-17).

