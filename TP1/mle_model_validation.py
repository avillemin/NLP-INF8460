"""
Questions 1.3.1 et 1.3.2 : validation de votre modèle avec NLTK

Dans ce fichier, on va comparer le modèle obtenu dans `mle_ngram_model.py` avec le modèle MLE fourni par NLTK.

Pour préparer les données avant d'utiliser le modèle NLTK, on pourra utiliser
>> ngrams, words = padded_everygram_pipeline(n, corpus)
>> vocab = Vocabulary(words, unk_cutoff=k)

Lors de l'initialisation d'un modèle NLTK, il faut passer une variable order qui correspond à l'ordre du modèle n-gramme,
et une variable vocabulary de type Vocabulary.

On peut ensuite entraîner le modèle avec la méthode model.fit(ngrams). Attention, la documentation prête à confusion :
la méthode attends une liste de liste de n-grammes (`list(list(tuple(str)))` et non pas `list(list(str))`).
"""
from nltk.lm.models import MLE
from nltk.lm.vocabulary import Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline
from preprocess_corpus import read_and_preprocess
from mle_ngram_model import extract_ngrams, NgramModel


def train_MLE_model(corpus, n):
    """
    Entraîne un modèle de langue n-gramme MLE de NLTK sur le corpus.

    :param corpus: list(list(str)), un corpus tokenizé
    :param n: l'ordre du modèle
    :return: un modèle entraîné
    """
    
    ngrams, words = padded_everygram_pipeline(n, corpus)
    vocab = Vocabulary(words, unk_cutoff=1)
    lm = MLE(n, vocabulary=vocab)
    lm.fit(ngrams)
    
    return lm


def compare_models(your_model, nltk_model, corpus, n):
    """
    Pour chaque n-gramme du corpus, calcule la probabilité que lui attribuent `your_model`et `nltk_model`, et
    vérifie qu'elles sont égales. Si un n-gramme a une probabilité différente pour les deux modèles, cette fonction
    devra afficher le n-gramme en question suivi de ses deux probabilités.

    À la fin de la comparaison, affiche la proportion de n-grammes qui diffèrent.

    :param your_model: modèle NgramModel entraîné dans le fichier 'mle_ngram_model.py'
    :param nltk_model: modèle nltk.lm.MLE entraîné sur le même corpus dans la fonction 'train_MLE_model'
    :param corpus: list(list(str)), une liste de phrases tokenizées à tester
    :return: float, la proportion de n-grammes incorrects
    """
    ngrams = extract_ngrams(corpus,n)
    nb_total_ngrams = 0
    nb_total_difference = 0
    for sentence in range(1):
        for ngram in ngrams[sentence]:
            nb_total_ngrams+=1
            proba_your_model = your_model.proba(ngram[-1],ngram[:-1])
            proba_nltk_model = nltk_model.score(ngram[-1],ngram[:-1])
            if proba_your_model != proba_nltk_model:
                print(ngram, proba_your_model, proba_nltk_model)
                nb_total_difference+=1
                
    return nb_total_difference/nb_total_ngrams
    

if __name__ == "__main__":
    """
    Ici, vous devrez valider votre implémentation de `NgramModel` en la comparant avec le modèle NLTK. Pour n=1, 2, 3,
    vous devrez entraîner un modèle nltk `MLE` et un modèle `NgramModel` sur `shakespeare_train`, et utiliser la fonction 
    `compare_models `pour vérifier qu'ils donnent les mêmes résultats. 
    Comme corpus de test, vous choisirez aléatoirement 50 phrases dans `shakespeare_train`.
    """

    filename = "./data/shakespeare_train.txt"
    preprocessed_corpus = read_and_preprocess(filename)
    
    nltk_model_1 = train_MLE_model(preprocessed_corpus, 1)
    my_model_1 = NgramModel(preprocessed_corpus, 1)
    
    nltk_model_2 = train_MLE_model(preprocessed_corpus, 2)
    my_model_2 = NgramModel(preprocessed_corpus, 2)
    
    nltk_model_3 = train_MLE_model(preprocessed_corpus, 3)
    my_model_3 = NgramModel(preprocessed_corpus, 3)
    
    print('Compare_models avec n=1')
    error_1 = compare_models(my_model_1, nltk_model_1, preprocessed_corpus, 1)
    print(error_1)
    print()
    print('Compare_models avec n=2')
    error_2 = compare_models(my_model_2, nltk_model_2, preprocessed_corpus, 2)
    print(error_2)
    print()
    print('Compare_models avec n=3')
    error_3 = compare_models(my_model_3, nltk_model_3, preprocessed_corpus, 3)
    print(error_3)
    
    pass
