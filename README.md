# Topic Modeling in Python for Social Sciences
Handy *Jupyter Notebooks*, *python scripts*, *mindmaps* and scientific literature that I use in for Topic Modeling. Including text mining from PDF files, text preprocessing, Latent Dirichlet Allocation (LDA), hyperparameters grid search and Topic Modeling visualiation.

## List of Notebooks
* textract-PDF: batch text mining from PDFs 
* text-pre-process: tokenization, stopwords and stemming from `spaCy`
* gensim-pre-process: tokenization, stopwords and stemming from `gensim`
* gensim-topics: LDA in `gensim`
* gensim-LDA-Mallet: LDA in `gensim` using a `MALLET` wrapper
* gensim-optimal-topics: choose the number of topics to give the highest coherence and perplexity values in LDA models in `gensim`
* gensim-topic-analysis: exploratory analysis of topics from LDA models in `gensim`
* scikit-learn-best-topic-model:  LDA in `scikit-learn` and optimal hyperparameter grid search; it also includes a pyLDAvis wrapper for `scikit-learn` for interactive LDA results visualization
* gensim-topic-modeling-visualization: multiple strategies to visualize the results of a `gensim` LDA model
* gensim-pyLDAvis: pyLDAvis wrapper for `gensim` for interactive LDA results visualization
* Stop-words Comparison: *english* and *portuguese* stopwords comparison from `NLTK`, `spaCy` and `gensim`

## List of python function in utils
* clean_up: Clean up you text and generate list of words for each document using `spaCy`.
* compute_performance: Generate a model list for number of topics and compute c_v coherence and perplexity (if applicable) using either `gensim.models.ldamodel.LdaModel`, `gensim.models.ldamulticore.LdaMulticore` or `gensim.models.wrappers.LdaMallet`. It enables visualization of the optimal topic number for your model.
* graph_performance: Graphics for visualizing the output of a compute performance for a LDA model list.

## Scientific Literature
A *BibTeX* file with the relevant collection of scientific literature (mainly articles published in peer-reviewed journals). It is comprised of computer science literature about topic modeling algorithms and procedures; and social science literature about best practices and uses of topic modeling along with interesting applications of topic modeling to textual data.

## Mind Maps
I tend to aggregate and organize my knowledge about a subject in mind maps. There are some mind maps about topic modeling as PDF files with some content already referenced with the relevant literature.

## Stopwords Comparison
As of June 15h 2019

|        	| English 	| Portuguese 	|
|--------	|:-------:	|:----------:	|
| spaCy  	|   326   	|     413    	|
| NLTK   	|   179   	|     203    	|
| gensim 	|   337   	|            	|

# Author
Jose Eduardo Storopoli