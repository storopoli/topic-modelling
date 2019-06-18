def graph_performance(model_list, coherence_values, perplexity_values=None, fname='performance LDA.jpg', dpi=300):
    """
    Graphics for visualizing the output of a compute performance for a LDA model list

    Parameters:
    ----------
    model_list : List of LDA topic models from a compute performance function
    coherence_values : Coherence values corresponding to the LDA models from a compute performance function
    perplexity_values : Perplexity values (if applicable) corresponding to the LDA models from a compute performance function
    fname : Filename to save figure
    dpi : Desired DPI for figure quality

    Returns:
    -------
    Saves a figure as a desired filename
    """
    import matplotlib.pyplot as plt
    x = range(2,2+len(model_list))
    
    if type(model_list[0]) == gensim.models.wrappers.ldamallet.LdaMallet:
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.savefig(fname=fname, dpi=dpi)
        plt.show()
        # Print the coherence scores
        count = 0
        for m, cv in zip(x, coherence_values):
            print('Model number: ', count, "Num Topics =", m, " has Coherence Value of", round(cv, 3))
            count += 1
            
    elif type(model_list[0]) == gensim.models.ldamodel.LdaModel:
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        plt.xlabel("Num Topics")
        plt.ylabel("Perplexity score", color=color)
        ax1.plot(x, perplexity_values, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Coherence score', color=color)  # we already handled the x-label with ax1
        ax2.plot(x, coherence_values, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(fname=fname, dpi=dpi)
        plt.show()
        # Print the performance scores
        count = 0
        for m, cv, per in zip(x, coherence_values, perplexity_values):
            print('Model number: ', count, "Num Topics =", m, 
              " has Coherence Value of", round(cv, 3),
             " and Perplexity Valye of", round(per, 3))
            count += 1