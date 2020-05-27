import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS

class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)
        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size * 2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # ...
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.training = training
        self.gru_definition = tf.keras.layers.GRU(self.hidden_size)
        self.gru_bidirectional = tf.keras.layers.Bidirectional(self.gru_definition, merge_mode='concat')
        ### TODO(Students) END


    def attn(self, rnn_result):
        ### TODO(Students) START
        # ...
        param = tf.nn.softmax(tf.tensordot(tf.tanh(rnn_result), self.omegas, axes=1, name='dotProduct'), name='alphas')
        output = tf.tanh(tf.reduce_sum(rnn_result * tf.expand_dims(param, -1), 0))
        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):
        word_embedding = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embedding = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        # ...
        word_pos_combined = tf.concat([word_embedding, pos_embedding], axis=2)
        masking = tf.cast(inputs != 0, tf.float32)
        rnn_result = self.gru_bidirectional(word_embedding, training=training, mask=masking)
        attention_val = self.attn(rnn_result)
        logits = self.decoder(attention_val)     
        ### TODO(Students) END

        return {'logits': logits}

class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 300, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        
        self.num_classes = len(ID_TO_CLASS)
        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size * 2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.training = training
        self.lstm_definition = tf.keras.layers.LSTM(self.hidden_size)
        self.lstm_bidirectional = tf.keras.layers.Bidirectional(self.lstm_definition, merge_mode='concat')
     
        self.final_dense = layers.Dense(self.embed_dim,activation='relu')
        self.layer_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense_1 = layers.Dense(self.embed_dim)
        self.dense_2 = layers.Dense(self.embed_dim)
        self.dense_3 = layers.Dense(self.embed_dim)

    def attn(self, rnn_result):
        param = tf.nn.softmax(tf.tensordot(tf.tanh(rnn_result), self.omegas, axes=1, name='dotProduct'), name='alphas')
        output = tf.tanh(tf.reduce_sum(rnn_result * tf.expand_dims(param, -1), 0))
        return output

    def new_attention_network(self,queries, keys, num_units, num_heads):
        layer_1 = tf.concat(tf.split(self.dense_1(queries), num_heads, axis=2), axis=0)
        layer_2 = tf.concat(tf.split(self.dense_2(keys), num_heads, axis=2), axis=0)
        layer_3 = tf.concat(tf.split(self.dense_3(keys), num_heads, axis=2), axis=0)
        result =  (tf.matmul(layer_1, tf.transpose(layer_2, [0, 2, 1]))) / (layer_2.get_shape().as_list()[-1] ** 0.5)
        result = tf.where(tf.equal(tf.tile(tf.expand_dims(tf.tile(tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))), [num_heads, 1]), 1), [1, tf.shape(queries)[1], 1]), 0), tf.ones_like(result) * (-2 ** 32 + 1), result)
        param = tf.nn.softmax(result)*(tf.tile(tf.expand_dims(tf.tile(tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))), [num_heads, 1]), -1), [1, 1, tf.shape(keys)[1]]))
        result = self.final_dense(tf.concat(tf.split(tf.matmul(param, layer_3), num_heads, axis=0), axis=2))
        result += queries
        result = self.layer_normalization(result)
        return result, param


    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)
        word_pos_combined = tf.concat([word_embed, pos_embed], axis=2)
        self_attn, self_alpha = self.new_attention_network(word_embed, word_embed, self.embed_dim, 4)
        logits = self.decoder(self.attn(self.lstm_bidirectional(self_attn, training=training, mask=tf.cast(inputs != 0, tf.float32))))

        return {'logits': logits}
