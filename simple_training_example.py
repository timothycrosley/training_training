"""
MINIMAL SENTIMENT ANALYSIS FROM SCRATCH
=====================================

A super simple sentiment analysis model that you can understand completely.
You provide your own data, train a basic neural network, and test it immediately.

What you'll learn:
1. How text becomes numbers
2. How neural networks learn patterns
3. How to measure if it's working

No complex libraries - just numpy and basic Python!
"""

import numpy as np
import re
from collections import Counter


# =============================================================================
# STEP 1: YOUR DATA - EDIT THIS!
# =============================================================================

# Add your own examples here!
# Format: (text, sentiment) where sentiment is 1 for positive, 0 for negative
YOUR_TRAINING_DATA = [
    # Positive examples (1)
    ("I love this movie", 1),
    ("This is awesome", 1),
    ("Great job", 1),
    ("I feel happy", 1),
    ("Amazing work", 1),
    ("So good", 1),
    ("I like it", 1),
    ("Wonderful experience", 1),
    # Negative examples (0)
    ("I hate this", 0),
    ("This is terrible", 0),
    ("Bad movie", 0),
    ("I feel sad", 0),
    ("Awful experience", 0),
    ("So bad", 0),
    ("I dislike it", 0),
    ("Horrible work", 0),
]

# Test data - add your own examples to test the trained model
YOUR_TEST_DATA = [
    ("I really love it", 1),
    ("This is very bad", 0),
    ("Great movie", 1),
    ("Terrible experience", 0),
]


# =============================================================================
# STEP 2: SIMPLE TEXT PROCESSING
# =============================================================================


def clean_text(text):
    """Convert text to lowercase and remove punctuation"""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Keep only letters and spaces
    return text.strip()


def tokenize(text):
    """Split text into individual words"""
    return clean_text(text).split()


def build_vocabulary(training_data):
    """
    Create a mapping from words to numbers.

    The neural network needs numbers, not words!
    So "love" might become 5, "hate" might become 12, etc.
    """
    all_words = []

    # Collect all words from all training texts
    for text, _ in training_data:
        words = tokenize(text)
        all_words.extend(words)

    # Count how often each word appears
    word_counts = Counter(all_words)

    # Create word-to-number mapping (vocabulary)
    vocabulary = {}
    for i, (word, count) in enumerate(word_counts.most_common()):
        vocabulary[word] = i

    print(f"📚 Vocabulary created with {len(vocabulary)} unique words:")
    print(f"   {list(vocabulary.keys())[:10]}...")  # Show first 10 words

    return vocabulary


def text_to_vector(text, vocabulary):
    """
    Convert text to a vector of numbers the neural network can understand.

    This creates a "bag of words" - we count how many times each word appears.
    Example: "I love love this" with vocab {"I":0, "love":1, "this":2}
    becomes [1, 2, 1] (I appears 1 time, love appears 2 times, this appears 1 time)
    """
    words = tokenize(text)
    vector = np.zeros(len(vocabulary))

    for word in words:
        if word in vocabulary:
            word_index = vocabulary[word]
            vector[word_index] += 1  # Count the word

    return vector


# =============================================================================
# STEP 3: SIMPLE NEURAL NETWORK
# =============================================================================


class SimpleNeuralNetwork:
    """
    A basic neural network with one hidden layer.

    Input Layer -> Hidden Layer (8 neurons) -> Output Layer (1 neuron)

    Think of it like: text numbers -> pattern detection -> positive/negative decision
    """

    def __init__(self, input_size):
        # Initialize with small random weights
        self.W1 = np.random.randn(input_size, 8) * 0.1  # Input to hidden
        self.b1 = np.zeros((1, 8))  # Hidden bias
        self.W2 = np.random.randn(8, 1) * 0.1  # Hidden to output
        self.b2 = np.zeros((1, 1))  # Output bias

        print(f"🧠 Neural network created!")
        print(f"   Input size: {input_size} (vocabulary size)")
        print(f"   Hidden layer: 8 neurons")
        print(f"   Output: 1 neuron (positive/negative)")

    def sigmoid(self, x):
        """Activation function - squashes numbers between 0 and 1"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    def forward(self, X):
        """
        Forward pass: calculate prediction

        X (text as numbers) -> hidden layer -> output (0-1 probability)
        """
        # Hidden layer calculation
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Output layer calculation
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, output):
        """
        Backward pass: calculate how to improve the weights

        This is where the "learning" happens - we figure out which weights
        need to change to make better predictions.
        """
        m = X.shape[0]  # Number of examples

        # Calculate gradients (how much to change each weight)
        dZ2 = output - y
        dW2 = (1 / m) * self.a1.T.dot(dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = dZ2.dot(self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = (1 / m) * X.T.dot(dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2, learning_rate):
        """Update the weights based on what we learned"""
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def predict(self, X):
        """Make predictions (0 or 1)"""
        probabilities = self.forward(X)
        return (probabilities > 0.5).astype(int)


# =============================================================================
# STEP 4: TRAINING FUNCTION
# =============================================================================


def train_model(training_data, epochs=1000, learning_rate=1.0):
    """
    Train the neural network on your data.

    This is where the magic happens - the network learns to associate
    certain words with positive or negative sentiment.
    """
    print(f"\n🏋️ Starting training with {len(training_data)} examples...")

    # Build vocabulary from your training data
    vocabulary = build_vocabulary(training_data)

    # Convert all training texts to number vectors
    X_train = []
    y_train = []

    for text, sentiment in training_data:
        vector = text_to_vector(text, vocabulary)
        X_train.append(vector)
        y_train.append(sentiment)

    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(-1, 1)

    print(f"📊 Training data shape: {X_train.shape}")
    print(f"   Each text is now a vector of {X_train.shape[1]} numbers")

    # Create and train the neural network
    model = SimpleNeuralNetwork(len(vocabulary))

    print(f"\n📈 Training for {epochs} epochs...")

    for epoch in range(epochs):
        # Forward pass: get predictions
        output = model.forward(X_train)

        # Calculate loss (how wrong we are)
        loss = np.mean((output - y_train) ** 2)

        # Backward pass: learn from mistakes
        dW1, db1, dW2, db2 = model.backward(X_train, y_train, output)

        # Update weights
        model.update_weights(dW1, db1, dW2, db2, learning_rate)

        # Print progress
        if epoch % 200 == 0:
            accuracy = np.mean((output > 0.5) == y_train) * 100
            print(f"   Epoch {epoch:4d}: Loss = {loss:.4f}, Accuracy = {accuracy:.1f}%")

    print(f"✅ Training complete!")

    return model, vocabulary


# =============================================================================
# STEP 5: TESTING AND PREDICTION
# =============================================================================


def test_model(model, vocabulary, test_data):
    """Test the trained model on new examples"""
    print(f"\n🧪 Testing model on {len(test_data)} examples...")

    correct = 0
    total = len(test_data)

    for text, true_sentiment in test_data:
        # Convert text to vector
        vector = text_to_vector(text, vocabulary)
        X_test = vector.reshape(1, -1)  # Reshape for single prediction

        # Get prediction
        probability = model.forward(X_test)[0][0]
        predicted_sentiment = 1 if probability > 0.5 else 0

        # Check if correct
        is_correct = predicted_sentiment == true_sentiment
        correct += is_correct

        # Show result
        sentiment_text = "😊 POSITIVE" if predicted_sentiment == 1 else "😞 NEGATIVE"
        confidence = probability if predicted_sentiment == 1 else (1 - probability)
        status = "✅" if is_correct else "❌"

        print(
            f'   {status} "{text}" -> {sentiment_text} (confidence: {confidence:.2f})'
        )

    accuracy = (correct / total) * 100
    print(f"\n📊 Test Accuracy: {accuracy:.1f}% ({correct}/{total})")

    return accuracy


def predict_sentiment(text, model, vocabulary):
    """Predict sentiment for any text you type"""
    vector = text_to_vector(text, vocabulary)
    X = vector.reshape(1, -1)

    probability = model.forward(X)[0][0]
    predicted_sentiment = 1 if probability > 0.5 else 0

    if predicted_sentiment == 1:
        return f"😊 POSITIVE (confidence: {probability:.2f})"
    else:
        return f"😞 NEGATIVE (confidence: {1 - probability:.2f})"


# =============================================================================
# STEP 6: MAIN EXECUTION
# =============================================================================


def main():
    """Main function - this is where everything happens!"""

    print("=" * 60)
    print("🎯 MINIMAL SENTIMENT ANALYSIS FROM SCRATCH")
    print("=" * 60)

    print(f"\n📝 Your training data ({len(YOUR_TRAINING_DATA)} examples):")
    for i, (text, sentiment) in enumerate(YOUR_TRAINING_DATA[:5]):  # Show first 5
        emoji = "😊" if sentiment == 1 else "😞"
        print(f'   {emoji} "{text}"')
    if len(YOUR_TRAINING_DATA) > 5:
        print(f"   ... and {len(YOUR_TRAINING_DATA) - 5} more examples")

    # Train the model
    model, vocabulary = train_model(YOUR_TRAINING_DATA)

    # Test the model
    test_accuracy = test_model(model, vocabulary, YOUR_TEST_DATA)

    # Interactive prediction
    print(f"\n🎮 Try your own examples! (type 'quit' to exit)")
    print("-" * 50)

    while True:
        user_text = input("\nEnter text to analyze: ").strip()

        if user_text.lower() in ["quit", "exit", "q"]:
            print("👋 Thanks for trying sentiment analysis!")
            break

        if user_text:
            result = predict_sentiment(user_text, model, vocabulary)
            print(f"Result: {result}")

    print(f"\n🎓 What you learned:")
    print(f"   • How text becomes numbers (vocabulary: {len(vocabulary)} words)")
    print(f"   • How neural networks find patterns (8 hidden neurons)")
    print(f"   • How to measure success (test accuracy: {test_accuracy:.1f}%)")
    print(f"   • The complete ML pipeline from data to predictions!")


if __name__ == "__main__":
    main()


# =============================================================================
# BONUS: UNDERSTANDING YOUR MODEL
# =============================================================================


def analyze_model(model, vocabulary):
    """
    BONUS: See what words the model thinks are positive/negative!

    This shows you which words the neural network has learned to associate
    with positive or negative sentiment.
    """
    print(f"\n🔍 BONUS: What did the model learn?")
    print("-" * 40)

    # Get the weights from input to hidden layer
    word_weights = model.W1.mean(axis=1)  # Average across all hidden neurons

    # Sort words by their weights
    word_importance = []
    for word, index in vocabulary.items():
        weight = word_weights[index]
        word_importance.append((word, weight))

    # Sort by weight (most positive to most negative)
    word_importance.sort(key=lambda x: x[1], reverse=True)

    print(f"Most POSITIVE words:")
    for word, weight in word_importance[:5]:
        print(f"   😊 {word}: {weight:.3f}")

    print(f"\nMost NEGATIVE words:")
    for word, weight in word_importance[-5:]:
        print(f"   😞 {word}: {weight:.3f}")

    print(
        f"\nThe neural network learned these patterns from your {len(YOUR_TRAINING_DATA)} examples!"
    )


# Uncomment this line to see what your model learned:
# analyze_model(model, vocabulary)
