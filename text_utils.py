import pickle

from HindiNLTK.normalization import DevanagariNormalizer
from HindiNLTK.pos_tagging.pos_tagging import POSTagger
from HindiNLTK.tokenizer import Tokenizer
from HindiNLTK.transliterate import Transliterate
from HindiNLTK.vectors.vectors import Vectorizer

stopwords = {'मैं', 'मुझको', 'मेरा', 'अपने', 'आप', 'को', 'हमने', 'हमारा', 'अपना', 'हम', 'आप', 'आपका', 'तुम्हारा',
             'अपने', 'आप', 'स्वयं', 'वह', 'इसे', 'उसके', 'खुद', 'को', 'कि', 'वह', 'उसकी', 'उसका', 'खुद', 'ही', 'यह',
             'इसके', 'उन्होने', 'अपने', 'क्या', 'जो', 'किसे', 'किसको', 'कि', 'ये', 'हूँ', 'होता', 'है', 'रहे', 'थी',
             'थे', 'होना', 'गया', 'किया', 'जा', 'रहा', 'है', 'किया', 'है', 'पडा', 'होने', 'करना', 'करता है', 'किया',
             'रही', 'एक', 'लेकिन', 'अगर', 'या', 'क्यूंकि', 'जैसा', 'जब', 'तक', 'जबकि', 'की', 'पर', 'द्वारा', 'के लिए',
             'साथ', 'के', 'बारे', 'में', 'खिलाफ', 'बीच', 'में', 'के', 'से', 'दौरान', 'सेे', 'के बाद', 'ऊपर', 'नीचे',
             'को', 'से', 'तक', 'से नीचे', 'करने में', 'निकल', 'बंद', 'से अधिक', 'तहत', 'दुबारा', 'आगे', 'फिर', 'एक बार',
             'यहाँ', 'वहाँ', 'कब', 'कहाँ', 'क्यों', 'कैसे', 'सारे', 'किसी', 'दोनो', 'प्रत्येक', 'ज्यादा', 'अधिकांश',
             'अन्य', 'में कुछ', 'ऐसा', 'में कोई', 'मात्र', 'खुद', 'समान', 'इसलिए', 'बहुत', 'सकता', 'जायेंगे', 'जरा',
             'चाहिए', 'अभी', 'और', 'कर दिया', 'रखें', 'का', 'हैं', 'इस', 'होता', 'करने', 'ने', 'बनी', 'तो', 'ही', 'हो',
             'इसका', 'था', 'हुआ', 'वाले', 'बाद', 'लिए', 'सकते', 'इसमें', 'दो', 'वे', 'करते', 'कहा', 'वर्ग', 'कई',
             'करें', 'होती', 'अपनी', 'उनके', 'यदि', 'हुई', 'जा', 'कहते', 'जब', 'होते', 'कोई', 'हुए', 'व', 'जैसे', 'सभी',
             'करता', 'उनकी', 'तरह', 'उस', 'आदि', 'इसकी', 'उनका', 'इसी', 'पे', 'तथा', 'भी', 'परंतु', 'इन', 'कम', 'दूर',
             'पूरे', 'गये', 'तुम', 'मै', 'यहां', 'हुये', 'कभी', 'अथवा', 'गयी', 'प्रति', 'जाता', 'इन्हें', 'गई', 'अब',
             'जिसमें', 'लिया', 'बड़ा', 'जाती', 'तब', 'उसे', 'जाते', 'लेकर', 'बड़े', 'दूसरे', 'जाने', 'बाहर', 'स्थान',
             'उन्हें ', 'गए', 'ऐसे', 'जिससे', 'समय', 'दोनों', 'किए', 'रहती', 'इनके', 'इनका', 'इनकी', 'सकती', 'आज', 'कल',
             'जिन्हें', 'जिन्हों', 'तिन्हें', 'तिन्हों', 'किन्हों', 'किन्हें', 'इत्यादि', 'इन्हों', 'उन्हों', 'बिलकुल',
             'निहायत', 'इन्हीं', 'उन्हीं', 'जितना', 'दूसरा', 'कितना', 'साबुत', 'वग़ैरह', 'कौनसा', 'लिये', 'दिया', 'जिसे',
             'तिसे', 'काफ़ी', 'पहले', 'बाला', 'मानो', 'अंदर', 'भीतर', 'पूरा', 'सारा', 'उनको', 'वहीं', 'जहाँ', 'जीधर',
             'के', 'एवं', 'कुछ', 'कुल', 'रहा', 'जिस', 'जिन', 'तिस', 'तिन', 'कौन', 'किस', 'संग', 'यही', 'बही', 'उसी',
             'मगर', 'कर', 'मे', 'एस', 'उन', 'सो', 'अत'}


def normalize(text):
    return DevanagariNormalizer().normalize(text)


def remove_stopwords(tokens):
    text = []
    for token in tokens:
        if token not in stopwords:
            text.append(token)
    return text


# # f = open("final_stopwords.txt", "r", encoding="utf-8")
# for x in f:
#   # print(x)
#   stopwords.append(x[0:-1])
def _generate_stem_words(word):
    suffixes = {
        1: [u"ो", u"े", u"ू", u"ु", u"ी", u"ि", u"ा"],
        2: [u"कर", u"ाओ", u"िए", u"ाई", u"ाए", u"ने", u"नी", u"ना", u"ते", u"ीं", u"ती", u"ता", u"ाँ", u"ां", u"ों",
            u"ें"],
        3: [u"ाकर", u"ाइए", u"ाईं", u"ाया", u"ेगी", u"ेगा", u"ोगी", u"ोगे", u"ाने", u"ाना", u"ाते", u"ाती", u"ाता",
            u"तीं", u"ाओं", u"ाएं", u"ुओं", u"ुएं", u"ुआं"],
        4: [u"ाएगी", u"ाएगा", u"ाओगी", u"ाओगे", u"एंगी", u"ेंगी", u"एंगे", u"ेंगे", u"ूंगी", u"ूंगा", u"ातीं",
            u"नाओं", u"नाएं", u"ताओं", u"ताएं", u"ियाँ", u"ियों", u"ियां"],
        5: [u"ाएंगी", u"ाएंगे", u"ाऊंगी", u"ाऊंगा", u"ाइयाँ", u"ाइयों", u"ाइयां"],
    }
    for L in 5, 4, 3, 2, 1:
        if len(word) > L + 1:
            for suf in suffixes[L]:
                # print type(suf),type(word),word,suf
                if word.endswith(suf):
                    # print 'h'
                    return word[:-L]
    return word


def generate_stem_dict(tokens):
    '''returns a dictionary of stem words for each token'''

    stem_word = {}
    for each_token in tokens:
        # print type(each_token)
        temp = _generate_stem_words(each_token)
        # print temp
        stem_word[each_token] = temp

    return stem_word


def generate_stem_list(tokens):
    stem_word = []
    for each_token in tokens:
        # print type(each_token)
        temp = _generate_stem_words(each_token)
        # print temp
        stem_word.append(temp)

    return stem_word


# print(stopwords)
t = Tokenizer()
l = (t.word_tokenize("इराक के विदेश मंत्री ने अमरीका के उस प्रस्ताव का मजाक उड़ाया है , जिसमें अमरीका ने संयुक्त राष्ट्र के प्रतिबंधों को इराकी नागरिकों के लिए कम हानिकारक बनाने के लिए कहा है ।")) #
tag = POSTagger()
print(tag.tag(l))
print(remove_stopwords(l))
print(t.sentence_tokenize("(इराक के विदेश मंत्री ने अमरीका के उस| प्रस्ताव का मजाक उड़ाया है , जिसमें अमरीका ने संयुक्त|"))
tr = Transliterate()
print(tr.transliterate(["Iraq", "mein", "videshi", "nagrik"]))
vectors = Vectorizer()
print(vectors.get_most_similar("इराक"))
# print(remove_stopwords(l))
# print(generate_stem_dict(l))
print(vectors.get_most_similar("राक"))
