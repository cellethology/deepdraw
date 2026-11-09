import gc

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

ONEHOT = np.vstack([np.eye(4, dtype=np.float32), np.zeros((1, 4), np.float32)])
COMPLEMENT = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


def build_model(length, outputs):
    inputs = layers.Input(shape=(length, 4))
    branches = []
    for width in (25, 11, 7):
        x = layers.Conv1D(600, width, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        branches.append(layers.Dropout(0.075)(x))
    x = layers.Concatenate()(branches)
    x = layers.Dense(64, activation="relu")(x)
    model = tf.keras.Model(inputs, layers.Dense(outputs)(x))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss="mse")
    return model


class DNABatches(tf.keras.utils.PyDataset):
    def __init__(
        self,
        codes,
        indices,
        targets=None,
        *,
        augment=False,
        shuffle=False,
        seed=0,
        batch_size=64,
    ):
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
        self.codes = codes
        self.indices = np.asarray(indices)
        self.targets = targets
        self.augment = augment
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.order = np.arange(len(indices) * (2 if augment else 1))
        self.on_epoch_end()

    def __len__(self):
        return (len(self.order) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, batch):
        order = self.order[batch * self.batch_size : (batch + 1) * self.batch_size]
        n = len(self.indices)
        codes = self.codes[self.indices[order % n]].copy()
        if self.augment:
            reverse = order >= n
            codes[reverse] = COMPLEMENT[codes[reverse, ::-1]]
        x = ONEHOT[codes]
        if self.targets is None:
            return (x,)
        return x, np.asarray(self.targets[order % n], dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def bootstrap_split(n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        train = rng.choice(n, n, replace=True)
        valid = np.setdiff1d(np.arange(n), train)
        if len(valid):
            return train, valid
    raise RuntimeError("Unable to draw a nonempty out-of-bag validation set")


def fit_predict(
    codes,
    observed,
    targets,
    remaining,
    seed,
    round_number,
    *,
    members=10,
    epochs=60,
    batch_size=64,
    prediction_batch_size=16,
):
    if min(members, epochs, batch_size, prediction_batch_size) < 1:
        raise ValueError("Training counts and minibatch sizes must be positive")
    predictions = np.zeros((len(remaining), targets.shape[1]), dtype=np.float64)
    for member in range(members):
        tf.keras.backend.clear_session()
        gc.collect()
        model_seed = int(
            np.random.SeedSequence([seed, round_number, 2000 + member]).generate_state(
                1
            )[0]
        )
        split_seed = np.random.SeedSequence([seed, round_number, 3000 + member])
        tf.keras.utils.set_random_seed(model_seed)
        train, valid = bootstrap_split(len(observed), split_seed)
        center = targets[train].mean(axis=0)
        training = DNABatches(
            codes,
            observed[train],
            targets[train] - center,
            augment=True,
            shuffle=True,
            seed=model_seed,
            batch_size=batch_size,
        )
        validation = DNABatches(
            codes,
            observed[valid],
            targets[valid] - center,
            augment=True,
            batch_size=batch_size,
        )
        model = build_model(codes.shape[1], targets.shape[1])
        history = model.fit(
            training,
            validation_data=validation,
            epochs=epochs,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-6,
                    patience=8,
                    restore_best_weights=True,
                )
            ],
        )
        if not all(np.isfinite(values).all() for values in history.history.values()):
            raise RuntimeError("Nonfinite CNN training loss")
        candidates = DNABatches(codes, remaining, batch_size=prediction_batch_size)
        predicted = np.asarray(model.predict(candidates, verbose=0), np.float64)
        if not np.isfinite(predicted).all():
            raise RuntimeError("Nonfinite CNN prediction")
        predictions += predicted + center
        del model, training, validation, candidates
    tf.keras.backend.clear_session()
    gc.collect()
    return predictions / members
