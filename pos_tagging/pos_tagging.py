import pickle


class POSTagger:

    def __init__(self):
        self.tagger = load_tagger()

    def tag(self, words):
        return self.tagger.tag(words)


def load_tagger():
    with open('pos_tagging/save.p', 'rb') as handle:
        m = pickle.load(handle)
    return m
