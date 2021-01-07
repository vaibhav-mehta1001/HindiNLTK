from typing import List

import sentencepiece as spm


class Tokenizer():

    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load("tokenizer_model.model")

    def tokenize(self, t: str) -> List[str]:

        l = self.sp.EncodeAsPieces(t)
        q = []
        for a in l:
            if a[0] == '▁':
                q.append(a[1:])
            else:
                q.append(a)
        return q