import pickle
import tensorflow.compat.v1 as tf

from HindiNLTK.tokenizer import Tokenizer

tf.disable_v2_behavior()


def _load_preprocess():
    with open('transliteration/preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)


def _load_params():
    with open('transliteration/params.p', mode='rb') as in_file:
        return pickle.load(in_file)


class Transliterate:

    def __init__(self):
        self.load_path = _load_params()
        self.batch_size = 30
        _, (self.source_vocab_to_int, self.target_vocab_to_int), (
            self.source_int_to_vocab, self.target_int_to_vocab) = _load_preprocess()

    # loading the saved parameters

    def transliterate(self, transliterate_words):
        # taking user input for prediction
        # initialising the graph
        words = []
        loaded_graph = tf.Graph()

        # initialising the session
        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph('transliteration/'+self.load_path + '.meta')

            # tf.train.Saver.restore(sess,load_path)
            loader.restore(sess, 'transliteration/'+self.load_path)

            # providing placeholder names from the loaded graph
            input_data = loaded_graph.get_tensor_by_name('input:0')
            logits = loaded_graph.get_tensor_by_name('predictions:0')
            target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
            keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

            for transliterate_word in transliterate_words:
                transliterate_word = _word_to_seq(transliterate_word, self.source_vocab_to_int)

                # transliterating the given word
                transliterate_logits = sess.run(logits, {input_data: [transliterate_word] * self.batch_size,
                                                     target_sequence_length: [len(
                                                         transliterate_word)] * self.batch_size,
                                                     keep_prob: 1.0})[0]
                # showing the output
                output = ""
                for i in transliterate_logits:
                    if self.target_int_to_vocab[i] != '<EOS>':
                        output = output + self.target_int_to_vocab[i]
                words.append(output)
            return words

# converting the words to vectors of integers
def _word_to_seq(word, vocab_to_int):
    results = []
    for word in list(word):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])

    return results
