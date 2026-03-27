import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# ==============================================================================
# 1. Transformer Components (Model Building Blocks)
# ==============================================================================

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
        )
        self.feed_forward = keras.Sequential([
            layers.Dense(d_ff, activation="relu"),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_layer = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        attn_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=mask,
            training=training,
        )
        x1 = self.norm1(inputs + attn_output)

        ffn_output = self.feed_forward(x1)
        ffn_output = self.dropout_layer(ffn_output, training=training)

        return self.norm2(x1 + ffn_output)


# ------------------------------------------------------------------------------

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

        position_indices = tf.range(start=0, limit=sequence_length, delta=1)
        angles = tf.range(start=0, limit=d_model, delta=1, dtype="float32")
        angles = 1 / tf.pow(10000.0, (angles - angles % 2) / d_model)

        position_indices = tf.cast(tf.expand_dims(position_indices, -1), "float32")
        angles = tf.cast(tf.expand_dims(angles, 0), "float32")

        positional_encoding = position_indices * angles
        positional_encoding = tf.where(tf.range(d_model) % 2 == 0,
                                       tf.sin(positional_encoding),
                                       tf.cos(positional_encoding))

        self.positional_encoding = tf.cast(positional_encoding, "float32")

    def call(self, inputs):
        return inputs + self.positional_encoding

# ------------------------------------------------------------------------------

class PredictionHead(keras.Model):
    def __init__(self, d_model, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(d_model // 2, activation="relu")
        self.dense2 = layers.Dense(output_dim, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# ------------------------------------------------------------------------------
# 2. Main Model
# ------------------------------------------------------------------------------

class IrrationalityIndex(keras.Model):
    def __init__(self, input_dim, d_model, seq_len, num_heads, d_ff, num_layers, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model

        self.factor_weights = layers.Dense(input_dim, activation='sigmoid')
        self.normalize = layers.Normalization(axis=-1)
        self.input_projection = layers.Dense(d_model)
        self.pos_embedding = PositionalEmbedding(seq_len, d_model)
        self.dropout = layers.Dropout(0.1)

        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, num_heads, d_ff, 0.1)
            for _ in range(num_layers)
        ]

        self.prediction_head = PredictionHead(d_model, output_dim)

    def call(self, inputs, training=False):
        inputs_scaled = self.normalize(inputs)
        weights = self.factor_weights(inputs_scaled)
        weighted_inputs = inputs_scaled * weights

        x = self.input_projection(weighted_inputs)
        x = self.pos_embedding(x)
        x = self.dropout(x, training=training)

        attention_mask = tf.reduce_all(tf.math.not_equal(inputs, 0), axis=-1)

        for block in self.encoder_blocks:
            x = block(x, training=training, mask=attention_mask)

        context_vector = x
        outputs = self.prediction_head(context_vector)

        # calculating index 0 to 100
        final_score = outputs * 100.0

        return final_score


# ------------------------------------------------------------------------------
# 3. Model Management: Saving and Loading
# ------------------------------------------------------------------------------

class ModelManager:
    def __init__(self, model, base_dir="Trained Models"):
        self.model = model
        self.base_dir = base_dir
        # create file directory
        tf.io.gfile.makedirs(self.base_dir)

    def save_weights(self, tag):
        """
        Saves trained weight
        tag --> file name
        """
        file_path = f"{self.base_dir}/weights_{tag}.h5"
        self.model.save_weights(file_path)
        print(f"Weights are successfully saved: {file_path}")
        return file_path

    def load_weights(self, tag):
        """
        loads trained weights.
        """
        file_path = f"{self.base_dir}/weights_{tag}.h5"
        try:
            self.model.load_weights(file_path)
            print(f"Weights are succussfully loaded: {file_path}")
            return True
        except tf.errors.NotFoundError:
            print(f"Error: {file_path} does not exist.")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    def save_model(self, tag):
        """
        saves both weights and model architecture.
        """
        model_path = f"{self.base_dir}/model_{tag}"
        self.model.save(model_path)
        print(f"Model is succussfully saved: {model_path}")
        return model_path

    @staticmethod
    def load_full_model(tag, base_dir="Trained Models"):
        """
        loads both weights and model architecture.
        """
        model_path = f"{base_dir}/model_{tag}"
        try:
            loaded_model = tf.keras.models.load_model(model_path)
            print(f"Model is successfully loaded: {model_path}")
            return loaded_model
        except Exception as e:
            print(f"Error: {e}")
            return None


# ------------------------------------------------------------------------------
# 4. Data Preparation and Training
# ------------------------------------------------------------------------------

class Index_Model:
    def create_sequences(self, data, target, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)].values)
            y.append(target.iloc[i + seq_length])
        return np.array(X), np.array(y)

    def prepare_data(self, data, target, seq_length, train_size=0.8, val_size=0.1):
        X, y = self.create_sequences(data, target, seq_length)

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        train_idx = int(len(X) * train_size)
        val_idx = int(len(X) * (train_size + val_size))

        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                mode='min'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                mode='min',
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
        ]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                tf.keras.metrics.MeanAbsoluteError(name='mae')
            ]
        )

        X_train = X_train.reshape(-1, model.seq_len, model.input_dim)
        X_val = X_val.reshape(-1, model.seq_len, model.input_dim)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history, model
