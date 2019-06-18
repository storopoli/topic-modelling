def clean_up(text):
    """
    This function clean up you text
    and generate list of words for 
    each document.
    
    It also corrects for unicode problems
    with python version 2.
    """
    import sys, spacy
    removal=['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE']
    text_out = []
    if sys.version_info.major == 2:
        text = unicode(''.join([i if ord(i) < 128 else ' ' for i in text]))
    doc = nlp(text)
    for token in doc:
        if token.is_stop == False and token.is_alpha and len(token)>2 and token.pos_ not in removal:
            lemma = token.lemma_
            text_out.append(lemma)
    return text_out