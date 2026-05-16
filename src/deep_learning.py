"""
THEORY MODULE 3: Deep Learning
Covers: ANN, CNN, RNN/LSTM, Autoencoders, Attention
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks


class DeepLearningTheory:
    """Complete deep learning implementation"""

    def __init__(self, input_dim):
        self.input_dim = input_dim

    def all_deep_learning_concepts(self):
        """Explain all DL concepts"""

        print("\n" + "=" * 70)
        print("🧠 DEEP LEARNING - COMPLETE THEORY")
        print("=" * 70)

        self._neural_network_basics()
        self._activation_functions()
        self._backpropagation()
        self._optimizers()
        self._regularization()
        self._autoencoders()
        self._lstm_for_fraud()

    def _neural_network_basics(self):
        """Neural network fundamentals"""
        print("\n1️⃣ NEURAL NETWORK BASICS")
        print("-" * 50)

        print("\n🏗️ ARCHITECTURE:")
        print("   Input Layer: 28 features (PCA components)")
        print("   Hidden Layer 1: 128 neurons")
        print("   Hidden Layer 2: 64 neurons")
        print("   Hidden Layer 3: 32 neurons")
        print("   Output Layer: 1 neuron (fraud probability)")

        print("\n🔢 NEURON COMPUTATION:")
        print("   z = Σ(wᵢxᵢ) + b")
        print("   a = activation(z)")
        print("   where:")
        print("     - w: weights (learned)")
        print("     - x: inputs (features)")
        print("     - b: bias")
        print("     - a: activation output")

        print("\n📊 FORWARD PROPAGATION:")
        print("   Input → Layer1 → Layer2 → Layer3 → Output")
        print("   Each layer transforms the data")

    def _activation_functions(self):
        """All activation functions"""
        print("\n2️⃣ ACTIVATION FUNCTIONS")
        print("-" * 50)

        functions = {
            'ReLU': {
                'formula': 'f(x) = max(0, x)',
                'use': 'Hidden layers (default)',
                'pros': 'No vanishing gradient, fast',
                'cons': 'Dead neurons for x<0'
            },
            'Sigmoid': {
                'formula': 'f(x) = 1/(1 + e^(-x))',
                'use': 'Binary classification output',
                'pros': 'Output between 0-1',
                'cons': 'Vanishing gradient'
            },
            'Tanh': {
                'formula': 'f(x) = (e^x - e^(-x))/(e^x + e^(-x))',
                'use': 'Hidden layers (RNN)',
                'pros': 'Zero-centered',
                'cons': 'Vanishing gradient'
            },
            'Softmax': {
                'formula': 'f(x_i) = e^(x_i)/Σe^(x_j)',
                'use': 'Multi-class output',
                'pros': 'Probability distribution',
                'cons': 'Computationally expensive'
            },
            'LeakyReLU': {
                'formula': 'f(x) = max(0.01x, x)',
                'use': 'Avoid dead neurons',
                'pros': 'No dead neurons',
                'cons': 'Extra parameter'
            }
        }

        for name, info in functions.items():
            print(f"\n📌 {name}")
            print(f"   Formula: {info['formula']}")
            print(f"   Best for: {info['use']}")

    def _backpropagation(self):
        """Backpropagation theory"""
        print("\n3️⃣ BACKPROPAGATION (How NNs Learn)")
        print("-" * 50)

        print("\n📐 CHAIN RULE:")
        print("   ∂L/∂w = ∂L/∂ŷ * ∂ŷ/∂z * ∂z/∂w")

        print("\n🔧 STEPS:")
        print("   1. Forward pass: Compute predictions")
        print("   2. Calculate loss (error)")
        print("   3. Backward pass: Compute gradients")
        print("   4. Update weights: w = w - η * ∂L/∂w")
        print("   5. Repeat for each batch")

        print("\n⚠️ CHALLENGES:")
        print("   Vanishing Gradient: Gradients become 0 (deep networks)")
        print("   Exploding Gradient: Gradients become huge")
        print("   Solutions: BatchNorm, Residual connections, Gradient clipping")

    def _optimizers(self):
        """Optimization algorithms"""
        print("\n4️⃣ OPTIMIZERS")
        print("-" * 50)

        optimizers = {
            'SGD': {
                'formula': 'w = w - η∇L(w)',
                'pros': 'Simple, memory efficient',
                'cons': 'Slow, stuck in local minima'
            },
            'Momentum': {
                'formula': 'v = βv + η∇L(w), w = w - v',
                'pros': 'Faster convergence',
                'cons': 'One extra hyperparameter'
            },
            'Adam': {
                'formula': 'Adaptive learning rates per parameter',
                'pros': 'Best default choice, fast',
                'cons': 'Can overfit'
            },
            'RMSprop': {
                'formula': 'Divide by running average of gradients',
                'pros': 'Good for RNNs',
                'cons': 'Learning rate decay'
            }
        }

        for name, info in optimizers.items():
            print(f"\n📌 {name}")
            print(f"   Formula: {info['formula']}")
            print(f"   Advantages: {info['pros']}")

    def _regularization(self):
        """Regularization techniques"""
        print("\n5️⃣ REGULARIZATION (Prevent Overfitting)")
        print("-" * 50)

        print("\n🎯 L1 REGULARIZATION (Lasso):")
        print("   Loss = Original Loss + λΣ|w|")
        print("   Effect: Creates sparse weights (feature selection)")

        print("\n🎯 L2 REGULARIZATION (Ridge):")
        print("   Loss = Original Loss + λΣw²")
        print("   Effect: Shrinks weights, no feature selection")

        print("\n🎯 DROPOUT:")
        print("   Randomly turn off neurons during training")
        print("   Forces network to be redundant")
        print("   Rate: 0.2-0.5 for hidden layers")

        print("\n🎯 BATCH NORMALIZATION:")
        print("   Normalize inputs to each layer")
        print("   Benefits:")
        print("     - Faster training")
        print("     - Higher learning rates")
        print("     - Less sensitive to initialization")

        print("\n🎯 EARLY STOPPING:")
        print("   Stop training when validation loss stops improving")
        print("   Patience: Wait N epochs before stopping")

    def _autoencoders(self):
        """Autoencoder for anomaly detection"""
        print("\n6️⃣ AUTOENCODERS")
        print("-" * 50)

        print("\n🎯 WHAT IT DOES:")
        print("   Learns to reconstruct input data")
        print("   Anomalies = high reconstruction error")

        print("\n🏗️ ARCHITECTURE:")
        print("   Input (28) → Encoder (16) → Latent (8) → Decoder (16) → Output (28)")

        print("\n🔧 HOW IT DETECTS FRAUD:")
        print("   1. Train on normal transactions only")
        print("   2. Autoencoder learns to reconstruct 'normal'")
        print("   3. For fraud: Reconstruction error is HIGH")
        print("   4. Threshold: Error > 95th percentile → FRAUD")

        print("\n💡 ADVANTAGES:")
        print("   ✅ No labeled fraud data needed")
        print("   ✅ Detects new/unknown fraud patterns")
        print("   ✅ Works with imbalanced data")

    def _lstm_for_fraud(self):
        """LSTM for sequence fraud detection"""
        print("\n7️⃣ LSTM (Long Short-Term Memory)")
        print("-" * 50)

        print("\n🎯 WHY LSTM FOR FRAUD:")
        print("   Fraud often happens in sequences")
        print("   Example: Multiple small transactions before big one")
        print("   LSTM remembers patterns over time")

        print("\n🏗️ LSTM CELL STRUCTURE:")
        print("   Input Gate: What to store")
        print("   Forget Gate: What to forget")
        print("   Output Gate: What to output")
        print("   Cell State: Long-term memory")

        print("\n📐 FORGET GATE:")
        print("   f_t = σ(W_f·[h_{t-1}, x_t] + b_f)")
        print("   Decides what to forget from previous state")

        print("\n📐 INPUT GATE:")
        print("   i_t = σ(W_i·[h_{t-1}, x_t] + b_i)")
        print("   C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)")
        print("   Decides what new info to store")

        print("\n📐 OUTPUT GATE:")
        print("   o_t = σ(W_o·[h_{t-1}, x_t] + b_o)")
        print("   h_t = o_t * tanh(C_t)")
        print("   Decides what to output")

    def build_ann_model(self):
        """Build complete ANN for fraud detection"""
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),

            # Hidden Layer 1
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Hidden Layer 2
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Hidden Layer 3
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),

            # Output Layer
            layers.Dense(1, activation='sigmoid')
        ])

        # Focal Loss for imbalanced data
        def focal_loss(gamma=2.0, alpha=0.25):
            def loss(y_true, y_pred):
                epsilon = tf.keras.backend.epsilon()
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

                pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
                alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)

                return -alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt)

            return loss

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=focal_loss(),
            metrics=['accuracy', keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()]
        )

        return model

    def build_autoencoder(self):
        """Build autoencoder for anomaly detection"""
        # Encoder
        encoder_input = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(64, activation='relu')(encoder_input)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.Dense(16, activation='relu')(encoded)
        encoder = models.Model(encoder_input, encoded, name='encoder')

        # Decoder
        decoder_input = layers.Input(shape=(16,))
        decoded = layers.Dense(32, activation='relu')(decoder_input)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        decoder = models.Model(decoder_input, decoded, name='decoder')

        # Autoencoder
        autoencoder_input = layers.Input(shape=(self.input_dim,))
        encoded_output = encoder(autoencoder_input)
        decoded_output = decoder(encoded_output)
        autoencoder = models.Model(autoencoder_input, decoded_output, name='autoencoder')

        autoencoder.compile(optimizer='adam', loss='mse')

        return autoencoder, encoder, decoder