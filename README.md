# Topic Modeling in Python
Handy *Jupyter Notebooks* that I use in for Topic Modeling. Including text mining from PDF files, text preprocessing, Latent Dirichlet Allocation (LDA), hyperparameters grid search and Topic Modeling visualiation.

## List of Notebooks
* textract-PDF: batch text mining from PDFs 
* text-pre-process: tokenization, stopwords and stemming from `spaCy`
* gensim-pre-process: tokenization, stopwords and stemming from `gensim`
* gensim-topics: LDA in `gensim`
* gensim-LDA-Mallet: LDA in `gensim` using a `MALLET` wrapper
* gensim-optimal-topics: choose the number of topics to give the highest coherence value in LDA models in `gensim`
* gensim-topic-analysis: exploratory analysis of topics from LDA models in `gensim`
* scikit-learn-best-topic-model:  LDA in `scikit-learn` and optimal hyperparameter grid search; it also includes a pyLDAvis wrapper for `scikit-learn` for interactive LDA results visualization
* gensim-topic-modeling-visualization: multiple strategies to visualize the results of a `gensim` LDA model
* gensim-pyLDAvis: pyLDAvis wrapper for `gensim` for interactive LDA results visualization
* Stop-words Comparison: *english* and *portuguese* stopwords comparison from `NLTK`, `spaCy` and `gensim`

## Stopwords Comparison
As of June 15h 2019

|        	| English 	| Portuguese 	|
|--------	|:-------:	|:----------:	|
| spaCy  	|   326   	|     413    	|
| NLTK   	|   179   	|     203    	|
| gensim 	|   337   	|            	|

# Author
Jose Eduardo Storopoli