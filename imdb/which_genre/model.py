from functools import partial

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TopKCategoricalAccuracy, Mean

from .params import fname_wordlist, categories

class GenreModel(Model):
    embed_dim = 10
    lstm_units = [30, 30, 30]
    target_dim = len(categories)

    def __init__(self):
        super().__init__()
        vocab = list(set(open(fname_wordlist).readlines()))
        self.vectorizer = TextVectorization(max_tokens=len(vocab), output_mode='int')
        self.vectorizer.adapt(vocab)
        self.emb = Embedding(len(vocab)+1, self.embed_dim)
        self.lstm1 = LSTM(self.lstm_units[0], return_sequences=True)
        self.lstm2 = LSTM(self.lstm_units[1], return_sequences=True)
        self.lstm3 = LSTM(self.lstm_units[2])
        self.out = Dense(self.target_dim, activation=sigmoid)

    def call(self, input, training=None, mask=None):
        x = self.vectorizer(input)
        x = self.emb(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        return self.out(x)

loss_object = BinaryCrossentropy()
optimizer = Adam()

accuracy = partial(TopKCategoricalAccuracy, k=2)
train_acc, test_acc = accuracy(name='train_acc'), accuracy(name='test_acc')

train_loss, test_loss = Mean(name='train_loss'), Mean(name='test_loss')
