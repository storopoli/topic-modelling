def compute_performance(dictionary, corpus, texts, limit, start=2, step=1, lda_type=None, mallet_path=None):
    """
    Compute c_v coherence and perplexity (if applicable) for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    lda_tpe : Type of LDA model from 3 options gensim for gensim.models.ldamodel.LdaModel, gensim-multicore for gensim.models.ldamulticore.LdaMulticore or mallet for gensim.models.wrappers.LdaMallet
    mallet_path : Full path of mallet-ver/bin/mallet

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    perplexity_values (if applicable) : Perplexity values corresponding to the LDA model with respective number of topics
    """
    import gensim
    coherence_values = []
    model_list = []
    perplexity_values = []
    if lda_type == 'gensim':
        for num_topics in range(start, limit, step):
            model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    num_topics=num_topics, 
                                                    id2word=dictionary,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)
            model_list.append(model)
            coherencemodel = gensim.models.coherencemodel.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
            perplexity = model.log_perplexity(corpus)
            perplexity_values.append(perplexity)

        return model_list, coherence_values, perplexity_values
    
    elif lda_type == 'gensim-multicore':
        for num_topics in range(start, limit, step):
            model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                                            num_topics=num_topics, 
                                                            id2word=dictionary,
                                                            random_state=100,
                                                            chunksize=100,
                                                            passes=10,
                                                            alpha='symmetric',
                                                            per_word_topics=True)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
            perplexity = model.log_perplexity(corpus)
            perplexity_values.append(perplexity)

        return model_list, coherence_values, perplexity_values

    elif lda_type == 'mallet':
        if mallet_path == None:
            raise Exception('mallet_path should be specified')
        else:
            for num_topics in range(start, limit, step):
                model = gensim.models.wrappers.LdaMallet(mallet_path, 
                                                         corpus=corpus, 
                                                         num_topics=num_topics, 
                                                         id2word=dictionary)
                model_list.append(model)
                coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    else:
        print('Please specify which lda_type: gensim, gensim-multicore or mallet')