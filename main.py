import tensorflow as tf
import numpy as np
from tqdm import tqdm  # For progress bar

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('derivative_model.h5')
print("Model loaded successfully.")

# Function to load and preprocess test data
def load_test_data(filename):
    inputs, ground_truths = [], []
    with open(filename, 'r') as f:
        for line in f:
            func, derivative = line.strip().split('=')
            func = func.split('(', 1)[1][:-1]  # Extract function
            inputs.append(func)
            ground_truths.append(derivative.strip())  # Extract derivative
    print(f"Loaded {len(inputs)} test samples.")
    return inputs, ground_truths

# Predict derivatives with a progress bar
def predict_derivatives(model, inputs, tokenizer, max_length=30):
    input_seqs = tokenizer.texts_to_sequences(inputs)
    input_seqs = tf.keras.preprocessing.sequence.pad_sequences(input_seqs, padding='post')

    predictions = []
    for seq in tqdm(input_seqs, desc="Generating predictions"):
        # Prepare the encoder input
        encoder_input = np.expand_dims(seq, axis=0)

        # Start decoding with an initial start token (assuming '1' is the start token index)
        decoder_input = np.zeros((1, 1))
        decoder_input[0, 0] = 1

        predicted_seq = []
        for _ in range(max_length):
            # Predict the next token
            output = model.predict([encoder_input, decoder_input], verbose=0)
            next_token = np.argmax(output[0, -1, :])

            if next_token == 0:  # End token or padding
                break

            predicted_seq.append(next_token)

            # Update decoder input with the new predicted token
            decoder_input = np.concatenate([decoder_input, [[next_token]]], axis=1)

        # Convert predicted sequence back to text
        predicted_text = ''.join([tokenizer.index_word.get(idx, '') for idx in predicted_seq])
        predictions.append(predicted_text)

    return predictions

# Load test data and ground truth derivatives
test_inputs, ground_truths = load_test_data('test.txt')

# Load tokenizer (same one used during training)
input_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
input_tokenizer.fit_on_texts(test_inputs)

# Generate predictions
print("Generating predictions...")
predictions = predict_derivatives(model, test_inputs, input_tokenizer)

# Calculate and print accuracy
correct = sum(1 for pred, truth in zip(predictions, ground_truths) if pred == truth)
accuracy = correct / len(ground_truths)

print(f"Accuracy: {accuracy * 100:.2f}%")
