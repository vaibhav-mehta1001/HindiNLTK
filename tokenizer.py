from typing import List

import sentencepiece as spm


class Tokenizer():

    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load("tokenizer_model.model")

    def tokenize(self, t: str) -> List[str]:
        l = self.sp.EncodeAsPieces(t)
        q = []
        for e in l:
            a = (e.replace('▁', ' '))
            if len(a) > 1:
                q.append(a[1:])
            else:
                q.append(a)
        return q

# t = Tokenizer()
# print(t.tokenize("ही माय मेंअमे || इस वैभव ।। //"))
