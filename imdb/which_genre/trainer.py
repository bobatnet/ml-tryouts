import click
import tensorflow as tf

from .load_data import data, sample_count, batch_size
from .model import GenreModel, loss_object, optimizer, train_acc, test_acc, train_loss, test_loss

model = GenreModel()

@tf.function
def train_step(input, target):
    with tf.GradientTape() as tape:
        pred = model(input, training=True)
        loss = loss_object(target, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_acc(pred, target)

@tf.function
def test_step(input, target):
    pred = model(input, training=False)
    loss = loss_object(target, pred)

    test_loss(loss)
    test_acc(pred, target)

@click.command()
@click.option('--epochs', default=5)
def run(epochs):
    """Train model for genre prediction from title"""

    for epoch in range(epochs):
        for metric in [train_acc, train_loss, test_loss, test_acc]:
            metric.reset_states()

        with click.progressbar(data, label='Train step', length=sample_count//batch_size) as data_:
            for inp, target in data_:
                train_step(inp, target)

        print(f'Epoch {epoch}\t, accuracy {train_acc.result()}')


if __name__ == '__main__':
    run()
