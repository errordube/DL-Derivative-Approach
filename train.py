import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Attention, Concatenate
from tensorflow.keras.models import Model
import numpy as np

# Function to load and preprocess the data
def load_data(filename):
    inputs, targets = [], []
    with open(filename, 'r') as f:
        for line in f:
            func, derivative = line.split('=')  # Split function from derivative
            input_expr = func.split('(', 1)[1][:-1]  # Extract expression
            target_expr = derivative.strip()  # Remove extra spaces
            inputs.append(input_expr)
            targets.append(target_expr)
    return inputs, targets

# Tokenizer function to convert expressions to sequences
def tokenize_data(data, tokenizer=None):
    if tokenizer is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
        tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    return sequences, tokenizer

# Load data
train_inputs, train_targets = load_data('train.txt')

# Tokenize input and target data
input_seqs, input_tokenizer = tokenize_data(train_inputs)
target_seqs, target_tokenizer = tokenize_data(train_targets)

# Pad sequences to the same length
input_seqs = tf.keras.preprocessing.sequence.pad_sequences(input_seqs, padding='post')
target_seqs = tf.keras.preprocessing.sequence.pad_sequences(target_seqs, padding='post')

# Define model architecture
input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_vocab_size, 64)(encoder_inputs)
encoder_lstm = LSTM(64, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(target_vocab_size, 64)(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention mechanism
attention = Attention()([decoder_outputs, encoder_outputs])
concat_attention = Concatenate()([decoder_outputs, attention])

# Output layer
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(concat_attention)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary and save it to network.txt
with open('network.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Prepare target data with start and end tokens for training
target_seqs_in = target_seqs[:, :-1]
target_seqs_out = target_seqs[:, 1:]

# Train the model
model.fit([input_seqs, target_seqs_in], np.expand_dims(target_seqs_out, -1), epochs=10, batch_size=64)

# Save the trained model
model.save('derivative_model.h5')
