from typing import List

import sentencepiece as spm


class Tokenizer():

    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load("tokenizer_model.model")

    def word_tokenize(self, t: str) -> List[str]:

        l = self.sp.EncodeAsPieces(t)
        q = []
        for a in l:
            if a[0] == '▁':
                q.append(a[1:])
            else:
                q.append(a)
        return q

    def sentence_tokenize(self, t: str):
        sentences = []
        sentence = ""
        i = 0
        for c in t:
            if c == "।" or c == '|' and i > 0:
                sentences.append(sentence)
                sentence = ""
            else:
                sentence += c
            i += 1
        return sentences
