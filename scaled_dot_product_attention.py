'''Author: Brandon Trabucco, Copyright 2019
Implements the scaled dot product attention mechanism from the transformer.
Vaswani, A. et al. https://arxiv.org/pdf/1706.03762.pdf'''


import tensorflow as tf
import collections


class ScaledDotProductAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, hidden_size, output_size, **kwargs):
        super(ScaledDotProductAttentionLayer, self).__init__(**kwargs)
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


class TransformerStateTuple(collections.namedtuple("TransformerStateTuple", ("sequences"))):
    __slots__ = ()
    @property
    def dtype(self):
        sequences = self[0]
        return sequences.dtype


class TransformerCell(tf.contrib.rnn.LayerRNNCell):

    def __init__(self, num_input_attention_layers, num_self_attention_layers, 
            num_input_attention_heads, num_self_attention_heads,
            input_attention_hidden_size, self_attention_hidden_size,
            input_attention_output_size, self_attention_output_size, 
            max_buffer_length=100, 
            input_attention_sequence=None, **kwargs):
        super(TransformerCell, self).__init__(**kwargs)
        self.input_attention_layers = [ScaledDotProductAttentionLayer(
            num_input_attention_heads, input_attention_hidden_size, 
            input_attention_output_size, **kwargs) for i in range(num_input_attention_layers)]
        self.self_attention_layers = [ScaledDotProductAttentionLayer(
            num_self_attention_heads, self_attention_hidden_size, 
            self_attention_output_size, **kwargs) for i in range(num_self_attention_layers)]
        self.joint_attention_layers = [ScaledDotProductAttentionLayer(
            num_self_attention_heads, self_attention_hidden_size, 
            self_attention_output_size, **kwargs) for i in range(num_self_attention_layers)]
        self.state_size = TransformerStateTuple([max_buffer_length, self_attention_output_size])
        self.output_size = output_size
        self.input_attention_sequence = input_attention_sequence

    def __call__(self, inputs, states):
        x = self.input_attention_sequence:
        for layer in self.input_attention_layers:
            x = tf.nn.relu(x + layer(x, x, x))
        next_states = tf.concat([states.sequences[:, 1:, :], tf.expand_dims(inputs, 1)], 1)
        y = next_states
        for layer1, layer2 in zip(self.self_attention_layers, self.joint_attention_layers):
            y = tf.nn.relu(y + layer1(y, y, y))
            y = tf.nn.relu(y + layer2(y, x, x))
        return tf.reduce_mean(y, 1), TransformerStateTuple(next_states)
    
    @property
    def trainable_variables(self):
        cell_variables = []
        for layer in self.input_attention_layers:
            cell_variables += layer.trainable_variables
        for layer in self.self_attention_layers:
            cell_variables += layer.trainable_variables
        for layer in self.joint_attention_layers:
            cell_variables += layer.trainable_variables
        return cell_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        cell_variables = []
        for layer in self.input_attention_layers:
            cell_variables += layer.variables
        for layer in self.self_attention_layers:
            cell_variables += layer.variables
        for layer in self.joint_attention_layers:
            cell_variables += layer.variables
        return cell_variables
    
    @property
    def weights(self):
        return self.variables
