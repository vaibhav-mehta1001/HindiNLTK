import sys, codecs, string, itertools, re
from os import name

import utils


class NormalizerI(object):
    """
    The normalizer classes do the following:
    * Some characters have multiple Unicode codepoints. The normalizer chooses a single standard representation
    * Some control characters are deleted
    * While typing using the Latin keyboard, certain typical mistakes occur which are corrected by the module
    Base class for normalizer. Performs some common normalization, which includes:
    * Byte order mark, word joiner, etc. removal
    * ZERO_WIDTH_NON_JOINER and ZERO_WIDTH_JOINER removal
    * ZERO_WIDTH_SPACE and NO_BREAK_SPACE replaced by spaces
    Script specific normalizers should derive from this class and override the normalize() method.
    They can call the super class 'normalize() method to avail of the common normalization
    """

    BYTE_ORDER_MARK = '\uFEFF'
    BYTE_ORDER_MARK_2 = '\uFFFE'
    WORD_JOINER = '\u2060'
    SOFT_HYPHEN = '\u00AD'

    ZERO_WIDTH_SPACE = '\u200B'
    NO_BREAK_SPACE = '\u00A0'

    ZERO_WIDTH_NON_JOINER = '\u200C'
    ZERO_WIDTH_JOINER = '\u200D'

    def normalize(self, text):
        pass


class BaseNormalizer(NormalizerI):

    def __init__(self, lang,
                 remove_nuktas=False,
                 nasals_mode='do_nothing',
                 do_normalize_chandras=False,
                 do_normalize_vowel_ending=False):

        self.lang = lang
        self.remove_nuktas = remove_nuktas
        self.nasals_mode = nasals_mode
        self.do_normalize_chandras = do_normalize_chandras
        self.do_normalize_vowel_ending = do_normalize_vowel_ending

        self._init_normalize_chandras()
        self._init_normalize_nasals()
        self._init_normalize_vowel_ending()
        # self._init_visarga_correction()

    def _init_normalize_vowel_ending(self):
        self.fn_vowel_ending = self._normalize_word_vowel_ending_ie

    def _init_normalize_chandras(self):

        substitution_offsets = \
            [
                [0x0d, 0x0f],  # chandra e, independent
                [0x11, 0x13],  # chandra o, independent
                [0x45, 0x47],  # chandra e , 0xde],pendent
                [0x49, 0x4b],  # chandra o , 0xde],pendent
                # [0x72 , 0x0f], # mr: chandra e, independent

                [0x00, 0x02],  # chandrabindu
                [0x01, 0x02],  # chandrabindu
            ]

        self.chandra_substitutions = [
            (utils.offset_to_char(x[0], self.lang), utils.offset_to_char(x[1], self.lang))
            for x in substitution_offsets]

    def _normalize_chandras(self, text):
        for match, repl in self.chandra_substitutions:
            text = text.replace(match, repl)
        return text

    def _init_to_anusvaara_strict(self):
        """
        `r1_nasal=re.compile(r'\\u0919\\u094D([\\u0915-\\u0918])')`
        """

        pat_signatures = \
            [
                [0x19, 0x15, 0x18],
                [0x1e, 0x1a, 0x1d],
                [0x23, 0x1f, 0x22],
                [0x28, 0x24, 0x27],
                [0x29, 0x24, 0x27],
                [0x2e, 0x2a, 0x2d],
            ]

        halant_offset = 0x4d
        anusvaara_offset = 0x02

        pats = []

        for pat_signature in pat_signatures:
            pat = re.compile(r'{nasal}{halant}([{start_r}-{end_r}])'.format(
                nasal=utils.offset_to_char(pat_signature[0], self.lang),
                halant=utils.offset_to_char(halant_offset, self.lang),
                start_r=utils.offset_to_char(pat_signature[1], self.lang),
                end_r=utils.offset_to_char(pat_signature[2], self.lang),
            ))
            pats.append(pat)

        repl_string = '{anusvaara}\\1'.format(anusvaara=utils.offset_to_char(anusvaara_offset, self.lang))

        self.pats_repls = (pats, repl_string)

    def _to_anusvaara_strict(self, text):

        pats, repl_string = self.pats_repls
        for pat in pats:
            text = pat.sub(repl_string, text)

        return text

    def _init_to_anusvaara_relaxed(self):
        """
        `r1_nasal=re.compile(r'\\u0919\\u094D([\\u0915-\\u0918])')`
        """

        nasals_list = [0x19, 0x1e, 0x23, 0x28, 0x29, 0x2e]
        nasals_list_str = ','.join([utils.offset_to_char(x, self.lang) for x in nasals_list])

        halant_offset = 0x4d
        anusvaara_offset = 0x02

        pat = re.compile(r'[{nasals_list_str}]{halant}'.format(
            nasals_list_str=nasals_list_str,
            halant=utils.offset_to_char(halant_offset, self.lang),
        ))

        repl_string = '{anusvaara}'.format(anusvaara=utils.offset_to_char(anusvaara_offset, self.lang))

        self.pats_repls = (pat, repl_string)

    def _to_anusvaara_relaxed(self, text):
        pat, repl_string = self.pats_repls
        return pat.sub(repl_string, text)

    def _init_to_nasal_consonants(self):
        """
        `r1_nasal=re.compile(r'\\u0919\\u094D([\\u0915-\\u0918])')`
        """

        pat_signatures = \
            [
                [0x19, 0x15, 0x18],
                [0x1e, 0x1a, 0x1d],
                [0x23, 0x1f, 0x22],
                [0x28, 0x24, 0x27],
                [0x29, 0x24, 0x27],
                [0x2e, 0x2a, 0x2d],
            ]

        halant_offset = 0x4d
        anusvaara_offset = 0x02

        pats = []
        repl_strings = []

        for pat_signature in pat_signatures:
            pat = re.compile(r'{anusvaara}([{start_r}-{end_r}])'.format(
                anusvaara=utils.offset_to_char(anusvaara_offset, self.lang),
                start_r=utils.offset_to_char(pat_signature[1], self.lang),
                end_r=utils.offset_to_char(pat_signature[2], self.lang),
            ))
            pats.append(pat)
            repl_string = '{nasal}{halant}\\1'.format(
                nasal=utils.offset_to_char(pat_signature[0], self.lang),
                halant=utils.offset_to_char(halant_offset, self.lang),
            )
            repl_strings.append(repl_string)

        self.pats_repls = list(zip(pats, repl_strings))

    def _to_nasal_consonants(self, text):

        for pat, repl in self.pats_repls:
            text = pat.sub(repl, text)

        return text

    def _init_normalize_nasals(self):

        if self.nasals_mode == 'to_anusvaara_strict':
            self._init_to_anusvaara_strict()
        elif self.nasals_mode == 'to_anusvaara_relaxed':
            self._init_to_anusvaara_relaxed()
        elif self.nasals_mode == 'to_nasal_consonants':
            self._init_to_nasal_consonants()

    def _normalize_nasals(self, text):
        if self.nasals_mode == 'to_anusvaara_strict':
            return self._to_anusvaara_strict(text)
        elif self.nasals_mode == 'to_anusvaara_relaxed':
            return self._to_anusvaara_relaxed(text)
        elif self.nasals_mode == 'to_nasal_consonants':
            return self._to_nasal_consonants(text)
        else:
            return text

    def _normalize_word_vowel_ending_dravidian(self, word):
        """
        for Dravidian
        - consonant ending: add 'a' ki maatra
        - halant ending: no change
        - 'a' ki maatra: no change
        """
        if len(word) > 0 and utils.is_consonant(word[-1], self.lang):
            return word + utils.offset_to_char(0x3e, self.lang)
        else:
            return word

    def _normalize_word_vowel_ending_ie(self, word):
        """
        for IE
        - consonant ending: add halant
        - halant ending: no change
        - 'a' ki maatra: no change
        """
        if len(word) > 0 and utils.is_consonant(word[-1], self.lang):
            return word + utils.offset_to_char(utils.HALANTA_OFFSET, self.lang)
        else:
            return word

    def _normalize_vowel_ending(self, text):
        return ' '.join([self.fn_vowel_ending(w) for w in text.split(' ')])

    def _normalize_puncutations(self, text):
        """
        Normalize punctuations.
        Applied many of the punctuation normalizations that are part of MosesNormalizer
        from sacremoses
        """
        text = text.replace(NormalizerI.BYTE_ORDER_MARK, '')
        text = text.replace('„', r'"')
        text = text.replace('“', r'"')
        text = text.replace('”', r'"')
        text = text.replace('–', r'-')
        text = text.replace('—', r' - ')
        text = text.replace('´', r"'")
        text = text.replace('‘', r"'")
        text = text.replace('‚', r"'")
        text = text.replace('’', r"'")
        text = text.replace("''", r'"')
        text = text.replace('´´', r'"')
        text = text.replace('…', r'...')

        return text

    def normalize(self, text):
        """
        Method to be implemented for normalization for each script
        """
        text = text.replace(NormalizerI.BYTE_ORDER_MARK, '')
        text = text.replace(NormalizerI.BYTE_ORDER_MARK_2, '')
        text = text.replace(NormalizerI.WORD_JOINER, '')
        text = text.replace(NormalizerI.SOFT_HYPHEN, '')

        text = text.replace(NormalizerI.ZERO_WIDTH_SPACE, ' ')  # ??
        text = text.replace(NormalizerI.NO_BREAK_SPACE, ' ')

        text = text.replace(NormalizerI.ZERO_WIDTH_NON_JOINER, '')
        text = text.replace(NormalizerI.ZERO_WIDTH_JOINER, '')

        text = self._normalize_puncutations(text)

        if self.do_normalize_chandras:
            text = self._normalize_chandras(text)
        text = self._normalize_nasals(text)
        if self.do_normalize_vowel_ending:
            text = self._normalize_vowel_ending(text)

        return text

    def get_char_stats(self, text):
        print(len(re.findall(NormalizerI.BYTE_ORDER_MARK, text)))
        print(len(re.findall(NormalizerI.BYTE_ORDER_MARK_2, text)))
        print(len(re.findall(NormalizerI.WORD_JOINER, text)))
        print(len(re.findall(NormalizerI.SOFT_HYPHEN, text)))

        print(len(re.findall(NormalizerI.ZERO_WIDTH_SPACE, text)))
        print(len(re.findall(NormalizerI.NO_BREAK_SPACE, text)))

        print(len(re.findall(NormalizerI.ZERO_WIDTH_NON_JOINER, text)))
        print(len(re.findall(NormalizerI.ZERO_WIDTH_JOINER, text)))

        # for mobj in re.finditer(NormalizerI.ZERO_WIDTH_NON_JOINER,text):
        #    print text[mobj.start()-10:mobj.end()+10].replace('\n', ' ').replace(NormalizerI.ZERO_WIDTH_NON_JOINER,'').encode('utf-8')
        # print hex(ord(text[mobj.end():mobj.end()+1]))

    def correct_visarga(self, text, visarga_char, char_range):
        text = re.sub(r'([\u0900-\u097f]):', '\\1\u0903', text)


class DevanagariNormalizer(BaseNormalizer):
    """
    Normalizer for the Devanagari script. In addition to basic normalization by the super class,
    * Replaces the composite characters containing nuktas by their decomposed form
    * replace pipe character '|' by poorna virama character
    * replace colon ':' by visarga if the colon follows a charcter in this script

    """

    NUKTA = '\u093C'

    def __init__(self, lang='hi', remove_nuktas=False, nasals_mode='do_nothing',
                 do_normalize_chandras=False, do_normalize_vowel_ending=False):
        super(DevanagariNormalizer, self).__init__(lang, remove_nuktas, nasals_mode, do_normalize_chandras,
                                                   do_normalize_vowel_ending)

    def normalize(self, text):
        # common normalization for Indic scripts
        text = super(DevanagariNormalizer, self).normalize(text)

        # chandra a replacement for Marathi
        text = text.replace('\u0972', '\u090f')

        # decomposing Nukta based composite characters
        text = text.replace('\u0929', '\u0928' + DevanagariNormalizer.NUKTA)
        text = text.replace('\u0931', '\u0930' + DevanagariNormalizer.NUKTA)
        text = text.replace('\u0934', '\u0933' + DevanagariNormalizer.NUKTA)
        text = text.replace('\u0958', '\u0915' + DevanagariNormalizer.NUKTA)
        text = text.replace('\u0959', '\u0916' + DevanagariNormalizer.NUKTA)
        text = text.replace('\u095A', '\u0917' + DevanagariNormalizer.NUKTA)
        text = text.replace('\u095B', '\u091C' + DevanagariNormalizer.NUKTA)
        text = text.replace('\u095C', '\u0921' + DevanagariNormalizer.NUKTA)
        text = text.replace('\u095D', '\u0922' + DevanagariNormalizer.NUKTA)
        text = text.replace('\u095E', '\u092B' + DevanagariNormalizer.NUKTA)
        text = text.replace('\u095F', '\u092F' + DevanagariNormalizer.NUKTA)

        if self.remove_nuktas:
            text = text.replace(DevanagariNormalizer.NUKTA, '')

        # replace pipe character for poorna virama
        text = text.replace('\u007c', '\u0964')

        # correct visarga
        text = re.sub(r'([\u0900-\u097f]):', '\\1\u0903', text)

        return text


# print("hi")
# n = DevanagariNormalizer()
# print(n.normalize("ही माय मेंअमे || इस वैभव ।। //"))
