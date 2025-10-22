import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def load_and_preprocess_data():
    """Load MNIST dataset and preprocess it."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Image shape: {x_train.shape[1:]}")

    return (x_train, y_train), (x_test, y_test)


def build_cnn_model():
    """Build a CNN model architecture for MNIST classification."""
    print("\nBuilding CNN model...")

    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Architecture:")
    model.summary()

    return model


def train_model(model, x_train, y_train, x_test, y_test):
    """Train the CNN model."""
    print("\nTraining model...")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=15,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    return history


def evaluate_model(model, x_test, y_test):
    """Evaluate the model on test data."""
    print("\nEvaluating model...")

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")

    return test_loss, test_accuracy


def visualize_predictions(model, x_test, y_test, num_samples=5):
    """Visualize model predictions on sample images."""
    print(f"\nVisualizing predictions on {num_samples} sample images...")

    indices = np.random.choice(len(x_test), num_samples, replace=False)

    sample_images = x_test[indices]
    sample_labels = y_test[indices]

    predictions = model.predict(sample_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i, ax in enumerate(axes):
        ax.imshow(sample_images[i].reshape(28, 28), cmap='gray')

        true_label = sample_labels[i]
        pred_label = predicted_labels[i]
        confidence = predictions[i][pred_label] * 100

        color = 'green' if true_label == pred_label else 'red'

        ax.set_title(
            f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
            color=color,
            fontsize=10
        )
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("Predictions saved to 'predictions.png'")
    plt.show()

    print("\nPrediction Details:")
    for i in range(num_samples):
        print(f"Sample {i+1}: True={sample_labels[i]}, "
              f"Predicted={predicted_labels[i]}, "
              f"Confidence={predictions[i][predicted_labels[i]]*100:.2f}%")


def main():
    """Main function to run the MNIST classifier."""
    print("=" * 60)
    print("MNIST Handwritten Digit Classifier using CNN")
    print("=" * 60)

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    model = build_cnn_model()

    history = train_model(model, x_train, y_train, x_test, y_test)

    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)

    if test_accuracy >= 0.95:
        print(f"\n✓ Target accuracy achieved: {test_accuracy*100:.2f}% (>95%)")
    else:
        print(f"\n✗ Target accuracy not met: {test_accuracy*100:.2f}% (<95%)")

    visualize_predictions(model, x_test, y_test, num_samples=5)

    model.save('mnist_cnn_model.keras')
    print("\nModel saved to 'mnist_cnn_model.keras'")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
