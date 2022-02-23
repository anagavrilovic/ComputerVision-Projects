from fuzzywuzzy import fuzz


def winner(output):
    # output je vektor sa izlaza neuronske mreze
    return max(enumerate(output), key=lambda x: x[1])[0]


def fuzzy_wuzzy(text, vocabulary):
    words = text.split()

    for word in words:
        if word == 'l' or word == 'i' or word == 't':
            switch_word = 'I'
            words[words.index(word)] = switch_word
            continue

        if vocabulary.get(word) is None:
            word_variations = [word, word.lower(), word.replace('t', 'l'), word.replace('g', 'y'), word.replace('o', 'e')]
            switch_word = next(iter(vocabulary))
            max_ratio = fuzz.ratio(word, switch_word)
            for key in vocabulary:
                for w in word_variations:
                    if fuzz.ratio(w, key) > max_ratio:
                        max_ratio = fuzz.ratio(w, key)
                        switch_word = key
            words[words.index(word)] = switch_word

    return " ".join(words)


def display_result(outputs, alphabet, k_means):
    """
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        k_means: obučen kmeans objekat
    Return:
        Vraća formatiran string
    """
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]

    return result
