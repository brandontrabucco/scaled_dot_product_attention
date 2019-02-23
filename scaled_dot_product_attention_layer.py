'''Author: Brandon Trabucco, Copyright 2019
Implements the scaled dot product attention mechamism from the transformer.
Vaswani, A. et al. https://arxiv.org/pdf/1706.03762.pdf'''


import tensorflow as tf


class ScaledDotProductAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, hidden_size, output_size, **kwargs):
        super(TransformerModule, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.query_map = tf.keras.layers.Dense(hidden_size * num_heads)
        self.key_map = tf.keras.layers.Dense(hidden_size * num_heads)
        self.value_map = tf.keras.layers.Dense(hidden_size * num_heads)
        self.output_map = tf.keras.layers.Dense(output_size)
    
    def __call__(self, queries, keys, values):
        batch_size, num_queries, sequence_length = tf.shape(queries)[0], tf.shape(queries)[1], tf.shape(values)[1]
        Q, K, V = self.query_map(queries), self.key_map(keys), self.value_map(values)
        Q = tf.transpose(tf.reshape(Q, [batch_size, num_queries, self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, [batch_size, sequence_length, self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, [batch_size, sequence_length, self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        S = tf.matmul(tf.nn.softmax(tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2])) / tf.sqrt(float(self.hidden_size))), V)
        S = tf.reshape(tf.transpose(S, [0, 2, 1, 3]), [batch_size, num_queries, self.num_heads * self.hidden_size])
        return self.output_map(S)  
        
    @property
    def trainable_variables(self):
        layer_variables = (
            self.query_map.trainable_variables + self.key_map.trainable_variables + 
            self.value_map.trainable_variables + self.output_map.trainable_variables )
        return layer_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        layer_variables = (
            self.query_map.variables + self.key_map.variables + 
            self.value_map.variables + self.output_map.variables )
        return layer_variables
    
    @property
    def weights(self):
        return self.variables
