import gensim

class Vectorizer:

    def __init__(self):
        self.__vector_model = gensim.models.KeyedVectors.load_word2vec_format("vectors/wiki.hi.vec", binary=False)

    def get_most_similar(self, word):
        return self.__vector_model.most_similar(word)

    def get_vector(self, word):
        return self.__vector_model.get_vector(word)
