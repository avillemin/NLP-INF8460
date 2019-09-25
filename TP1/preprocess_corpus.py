"""
Questions 1.1.1 à 1.1.5 : prétraitement des données.
"""

import nltk

def segmentize(raw_text):
    """
    Segmente un texte en phrases.

    >> raw_corpus = "Alice est là. Bob est ici"
    >> segmentize(raw_corpus)
    ["Alice est là.", "Bob est ici"]

    :param raw_text: str
    :return: list(str)
    """
    
    segmented_corpus = nltk.sent_tokenize(raw_text)
    
    return segmented_corpus


def tokenize(sentences):
    """
    Tokenize une liste de phrases en mots.

    >> sentences = ["Alice est là", "Bob est ici"]
    >> corpus = tokenize(sentences)
    >> corpus
    [
        ["Alice", "est", "là"],
        ["Bob", "est", "ici"]
    ]

    :param sentences: list(str), une liste de phrases
    :return: list(list(str)), une liste de phrases tokenizées
    """
    list_words_tokens = []
    for sentence in sentences:
    	list_words_tokens.append(nltk.word_tokenize(sentence))
    
    return list_words_tokens


def lemmatize(corpus):
    """
    Lemmatise les mots d'un corpus.

    :param corpus: list(list(str)), une liste de phrases tokenizées
    :return: list(list(str)), une liste de phrases lemmatisées
    """
    lemmzer = nltk.WordNetLemmatizer()
    lemmatized_corpus = []
    for sentence in corpus : 
        lemmatized_sentence = [lemmzer.lemmatize(token) for token in sentence]
        lemmatized_corpus.append(lemmatized_sentence)
    
    return lemmatized_corpus


def stem(corpus):
    """
    Retourne la racine (stem) des mots d'un corpus.

    :param corpus: list(list(str)), une liste de phrases tokenizées
    :return: list(list(str)), une liste de phrases stemées
    """
    
    stemmer = nltk.PorterStemmer()
    stemmed_corpus = []
    for sentence in corpus : 
        stemmed_sentence = [stemmer.stem(token) for token in sentence]
        stemmed_corpus.append(stemmed_sentence)
    
    return stemmed_corpus


def read_and_preprocess(filename):
    """
    Lit un fichier texte, puis lui applique une segmentation et une tokenization.

    [Cette fonction est déjà complète, on ne vous demande pas de l'écrire]

    :param filename: str, nom du fichier à lire
    :return: list(list(str))
    """
    with open(filename, "r") as f:
        raw_text = f.read()
    return tokenize(segmentize(raw_text))


def test_preprocessing(raw_text, sentence_id=0):
    """
    Applique à `raw_text` les fonctions segmentize, tokenize, lemmatize et stem, puis affiche le résultat de chacune
    de ces fonctions à la phrase d'indice `sentence_id`

    >> trump = open("data/trump.txt", "r").read()
    >> test_preprocessing(trump)
    Today we express our deepest gratitude to all those who have served in our armed forces.
    Today,we,express,our,deepest,gratitude,to,all,those,who,have,served,in,our,armed,forces,.
    Today,we,express,our,deepest,gratitude,to,all,those,who,have,served,in,our,armed,force,.
    today,we,express,our,deepest,gratitud,to,all,those,who,have,serv,in,our,arm,forc,.

    :param raw_text: str, un texte à traiter
    :param sentence_id: int, l'indice de la phrase à afficher
    :return: un tuple (sentences, tokens, lemmas, stems) qui contient le résultat des quatre fonctions appliquées à
    tout le corpus
    """
    segmentized_corpus = segmentize(raw_text)
    print(segmentized_corpus[sentence_id])
    tokenized_corpus = tokenize(segmentized_corpus)
    print(' '.join(tokenized_corpus[sentence_id]))
    lemmatized_corpus = lemmatize(tokenized_corpus)
    print(' '.join(lemmatized_corpus[sentence_id]))
    stemmed_corpus = stem(lemmatized_corpus)
    print(' '.join(stemmed_corpus[sentence_id]))
    return (segmentized_corpus, tokenized_corpus,lemmatized_corpus, stemmed_corpus)


if __name__ == "__main__":
    """
    Appliquez la fonction `test_preprocessing` aux corpus `shakespeare_train` et `shakespeare_test`.

    Note : ce bloc de code ne sera exécuté que si vous lancez le script directement avec la commande :
    ```
    python preprocess_corpus.py
    ```
    """
    filenames = ["./data/shakespeare_test", "./data/shakespeare_train"]
    
    for filename in filenames : # On repete le preprocess sur les deux corpus (train et test)
        with open(filename + ".txt", "r") as f:
            corpus = f.read()
        segmented_corpus = segmentize(corpus)

        with open('./output' + filename[6:] + "_phrases.txt", "w") as f:
            for phrase in segmented_corpus:
                f.write(phrase+'\n')

        tokenized_corpus = tokenize(segmented_corpus)

        with open('./output' + filename[6:] + "_mots.txt", "w") as f:
            for phrase in segmented_corpus:
                for word in phrase:
                    f.write(word+' ')
            f.write('\n')

        lemmatized_corpus = lemmatize(tokenized_corpus)
    	
        with open('./output' + filename[6:] + "_lemmes.txt","w") as f:
          	for phrase in lemmatized_corpus :
                  for word in phrase :
                      f.write(word+' ')
                  f.write('\n')
                
        stemmed_corpus = stem(lemmatized_corpus)
        
        with open('./output' + filename[6:] + "_stems.txt","w") as f:
            for phrase in stemmed_corpus :
                for word in phrase :
                    f.write(word+' ')
                f.write('\n')

        test_preprocessing(corpus,5)