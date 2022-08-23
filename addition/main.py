import sys

import keras
import numpy
from keras import layers
from keras import optimizers, losses, metrics

LEN_INPUTS = 5  # We are adding five single-digit numbers.
LEN_OUTPUTS = 46  # Sums of five single-digit numbers range from 0 to 45, giving 46 possible outputs.

# There are 10^5 i.e. 100k possible combinations. We train on half of them.
LEN_INPUTS_TRAIN = 50_000
LEN_INPUTS_VALIDATE = 10_000
LEN_INPUTS_TEST = 10_000
LEN_TOTAL_INPUTS = LEN_INPUTS_TRAIN + LEN_INPUTS_VALIDATE + LEN_INPUTS_TEST


def learn():
    """A neural network that learns to add five single-digit numbers."""

    input_layer = keras.Input(shape=(LEN_INPUTS,), name="inputs")
    hidden_layer = layers.Dense(64, activation="relu", name="hidden")(input_layer)
    outputs = layers.Dense(LEN_OUTPUTS, activation="softmax", name="outputs")(hidden_layer)
    model = keras.Model(inputs=input_layer, outputs=outputs)

    # We want to generate sufficient *unique* inputs, such that an addition only occurs once across the train,
    # validation and test datasets.
    inputs = numpy.random.randint(10, size=(LEN_TOTAL_INPUTS * 3, LEN_INPUTS))
    unique_inputs = numpy.unique(inputs, axis=0)
    if len(unique_inputs) < LEN_TOTAL_INPUTS:
        print(f'failed to generate {LEN_TOTAL_INPUTS} unique inputs')
        sys.exit(1)
    numpy.random.shuffle(unique_inputs)

    inputs_train = unique_inputs[:LEN_INPUTS_TRAIN]
    inputs_validate = unique_inputs[LEN_INPUTS_TRAIN:LEN_INPUTS_VALIDATE]
    inputs_test = unique_inputs[LEN_INPUTS_VALIDATE:LEN_TOTAL_INPUTS]

    # We sum across the digits in each sample to get the labels.
    labels_train = inputs_train.sum(1)
    labels_validate = inputs_validate.sum(1)
    labels_test = inputs_test.sum(1)

    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.SparseCategoricalCrossentropy()
    )

    print("\nFitting model on training data")
    model.fit(
        inputs_train,
        labels_train,
        batch_size=64,
        epochs=100,
        validation_data=(inputs_validate, labels_validate)
    )

    print("\nEvaluating on test data")
    results = model.evaluate(inputs_test, labels_test, batch_size=128)
    print("Test loss:", results)

    print("\nGenerating predictions")
    predictions = model.predict(inputs_test[:3])

    print("Predictions for first three samples:")
    for testcase, prediction in zip(inputs_test[:3], predictions):
        print(f"Input: {testcase}; prediction: {numpy.argmax(prediction)}")


if __name__ == '__main__':
    learn()
