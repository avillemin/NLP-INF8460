"""
Questions 1.1.6 à 1.1.8 : calcul de différentes statistiques sur un corpus.

Sauf mention contraire, toutes les fonctions renvoient un nombre (int ou float).
Dans toutes les fonctions de ce fichier, le paramètre `corpus` désigne une liste de phrases tokenizées, par exemple :
>> corpus = [
    ["Alice", "est", "là"],
    ["Bob", "est", "ici"]
]
"""
import preprocess_corpus as pre
import itertools


def count_tokens(corpus):
    count = 0
    """
    Renvoie le nombre de mots dans le corpus
    """
    for sentence in corpus:
        for word in sentence:
            count = count + 1
    return count



def count_types(corpus):
    """
    Renvoie le nombre de types (mots distincts) dans le corpus
    """
    words = []
    for sentence in corpus:
        for word in sentence:
            if word not in words:
                words.append(word)
    return len(words)



def get_most_frequent(corpus, n):
    from operator import itemgetter
    dictionnary = []
    most_frequent_pair_word_count = []
    for sentence in corpus:
        for element in sentence:
            dictionnary.append((element,sentence.count(element)))
    dictionnary =[(sum(i[1] for i in group), key)for key, group in itertools.groupby(sorted(dictionnary, key = lambda i: i[0]), lambda i: i[0])]
    dictionnary.sort(key=itemgetter(0),reverse=True)
    index =0
    while index < n :
        most_frequent_pair_word_count.append(dictionnary[index])
        index = index + 1
    for pair in most_frequent_pair_word_count:
        temp_pair = list(pair)
        temp_pair[0] = temp_pair[0]/count_tokens(corpus)
        pair = tuple(temp_pair)
    return most_frequent_pair_word_count
    """
    Renvoie les n mots les plus fréquents dans le corpus, ainsi que leurs fréquences

    :return: list(tuple(str, float)), une liste de paires (mot, fréquence)
    """


def get_token_type_ratio(corpus):
    """
    Renvoie le ratio nombre de tokens sur nombre de types
    """
    return count_tokens(corpus)/count_types(corpus)


def count_lemmas(corpus):
    lemmed_corpus = pre.lemmatize(corpus)
    lemma_count = count_types(lemmed_corpus)
    """
    Renvoie le nombre de lemmes distincts
    """
    return lemma_count


def count_stems(corpus):
    """
    Renvoie le nombre de racines (stems) distinctes
    """
    stemmed_corpus = pre.stem(corpus)
    stem_count = count_types(stemmed_corpus)
    return stem_count
    

def explore(corpus):
    """
    Affiche le résultat des différentes fonctions ci-dessus.

    Pour `get_most_frequent`, prenez n=15

    >> explore(corpus)
    Nombre de tokens: 5678
    Nombre de types: 890
    ...
    Nombre de stems: 650

    (Les chiffres ci-dessus sont indicatifs et ne correspondent pas aux résultats attendus)
    """
    print("Nombre de tokens: " + str(count_tokens(corpus)) + '\n')
    print("Nombre de types: " + str(count_types(corpus))+ '\n')
    print("Les 15 mots les plus frequents du vocabulaire: " + str(get_most_frequent(corpus,15)) + '\n')
    print("Le ratio est de :" + str(get_token_type_ratio(corpus)) + '\n')
    print("Le nombre de stems est de :" + str(count_stems(corpus)) + '\n')
    print("Le nombre de lemmes est de :" + str(count_lemmas(corpus)) + '\n')


if __name__ == "__main__":
    """
    Ici, appelez la fonction `explore` sur `shakespeare_train` et `shakespeare_test`. Quand on exécute le fichier, on 
    doit obtenir :

    >> python explore_corpus
    -- shakespeare_train --
    Nombre de tokens: 5678
    Nombre de types: 890
    ...

    -- shakespeare_test --
    Nombre de tokens: 78009
    Nombre de types: 709
    ...
    """
    filenames = ["./data/shakespeare_test", "./data/shakespeare_train"]
    for filename in filenames : 
        with open(filename + ".txt", "r") as f:
            corpus = f.read()
        segmented_corpus = pre.segmentize(corpus)
        tokenized_corpus = pre.tokenize(segmented_corpus)
        explore(tokenized_corpus)
