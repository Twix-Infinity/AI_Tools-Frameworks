import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import time


st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)


@st.cache_resource
def load_data():
    """Load and cache MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)


def build_cnn_model():
    """Build a CNN model architecture for MNIST classification."""
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

    return model


def visualize_samples(x_test, y_test, predictions=None, num_samples=5):
    """Create visualization of sample images with predictions."""
    indices = np.random.choice(len(x_test), num_samples, replace=False)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i, ax in enumerate(axes):
        img = x_test[indices[i]].reshape(28, 28)
        ax.imshow(img, cmap='gray')

        true_label = y_test[indices[i]]

        if predictions is not None:
            pred_probs = predictions[indices[i]]
            pred_label = np.argmax(pred_probs)
            confidence = pred_probs[pred_label] * 100
            color = 'green' if true_label == pred_label else 'red'

            ax.set_title(
                f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                color=color,
                fontsize=10
            )
        else:
            ax.set_title(f'Label: {true_label}', fontsize=10)

        ax.axis('off')

    plt.tight_layout()
    return fig


st.title("🔢 MNIST Handwritten Digit Classifier")
st.markdown("Train a CNN model to classify handwritten digits with >95% accuracy")

st.divider()

with st.sidebar:
    st.header("⚙️ Model Configuration")

    epochs = st.slider("Training Epochs", 5, 20, 15, 1)
    batch_size = st.selectbox("Batch Size", [64, 128, 256], index=1)

    st.divider()

    st.header("📊 Dataset Info")
    if st.button("Load Dataset"):
        with st.spinner("Loading MNIST dataset..."):
            (x_train, y_train), (x_test, y_test) = load_data()
            st.session_state.data_loaded = True
            st.session_state.x_train = x_train
            st.session_state.y_train = y_train
            st.session_state.x_test = x_test
            st.session_state.y_test = y_test
            st.success("Dataset loaded!")

if 'data_loaded' in st.session_state and st.session_state.data_loaded:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Training Samples", f"{len(st.session_state.x_train):,}")
    with col2:
        st.metric("Test Samples", f"{len(st.session_state.x_test):,}")
    with col3:
        st.metric("Image Shape", "28x28")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📚 Dataset Preview", "🏋️ Train Model", "🎯 Results"])

    with tab1:
        st.subheader("Sample Images from Dataset")

        if st.button("Show Random Samples"):
            fig = visualize_samples(
                st.session_state.x_test,
                st.session_state.y_test,
                num_samples=5
            )
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.subheader("Model Training")

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("🚀 Start Training", type="primary"):
                st.session_state.training_started = True

                progress_bar = st.progress(0)
                status_text = st.empty()

                with st.spinner("Building model..."):
                    model = build_cnn_model()
                    st.session_state.model = model

                status_text.text("Training in progress...")

                metrics_placeholder = st.empty()

                history_data = {
                    'epoch': [],
                    'accuracy': [],
                    'val_accuracy': [],
                    'loss': [],
                    'val_loss': []
                }

                class StreamlitCallback(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)

                        history_data['epoch'].append(epoch + 1)
                        history_data['accuracy'].append(logs['accuracy'])
                        history_data['val_accuracy'].append(logs['val_accuracy'])
                        history_data['loss'].append(logs['loss'])
                        history_data['val_loss'].append(logs['val_loss'])

                        metrics_placeholder.dataframe({
                            'Epoch': history_data['epoch'],
                            'Accuracy': [f"{a:.4f}" for a in history_data['accuracy']],
                            'Val Accuracy': [f"{a:.4f}" for a in history_data['val_accuracy']],
                            'Loss': [f"{l:.4f}" for l in history_data['loss']],
                            'Val Loss': [f"{l:.4f}" for l in history_data['val_loss']]
                        }, use_container_width=True)

                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True
                )

                history = model.fit(
                    st.session_state.x_train,
                    st.session_state.y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(st.session_state.x_test, st.session_state.y_test),
                    callbacks=[early_stopping, StreamlitCallback()],
                    verbose=0
                )

                st.session_state.history = history
                st.session_state.model_trained = True

                progress_bar.progress(1.0)
                status_text.text("Training complete!")

                st.success("✅ Model trained successfully!")

        with col2:
            st.info("**Model Architecture:**\n\n- Conv2D (32 filters) × 2\n- Conv2D (64 filters) × 2\n- Dense (256 units)\n- Output (10 classes)")

    with tab3:
        if 'model_trained' in st.session_state and st.session_state.model_trained:
            st.subheader("Model Performance")

            test_loss, test_accuracy = st.session_state.model.evaluate(
                st.session_state.x_test,
                st.session_state.y_test,
                verbose=0
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Test Accuracy", f"{test_accuracy*100:.2f}%")
            with col2:
                st.metric("Test Loss", f"{test_loss:.4f}")
            with col3:
                target_met = test_accuracy >= 0.95
                st.metric(
                    "Target (>95%)",
                    "✅ Met" if target_met else "❌ Not Met"
                )

            st.divider()

            st.subheader("Training History")

            col1, col2 = st.columns(2)

            with col1:
                fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
                ax_acc.plot(st.session_state.history.history['accuracy'], label='Training')
                ax_acc.plot(st.session_state.history.history['val_accuracy'], label='Validation')
                ax_acc.set_xlabel('Epoch')
                ax_acc.set_ylabel('Accuracy')
                ax_acc.set_title('Model Accuracy')
                ax_acc.legend()
                ax_acc.grid(True, alpha=0.3)
                st.pyplot(fig_acc)
                plt.close()

            with col2:
                fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
                ax_loss.plot(st.session_state.history.history['loss'], label='Training')
                ax_loss.plot(st.session_state.history.history['val_loss'], label='Validation')
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss')
                ax_loss.set_title('Model Loss')
                ax_loss.legend()
                ax_loss.grid(True, alpha=0.3)
                st.pyplot(fig_loss)
                plt.close()

            st.divider()

            st.subheader("Sample Predictions")

            if st.button("Generate Predictions"):
                predictions = st.session_state.model.predict(
                    st.session_state.x_test,
                    verbose=0
                )

                fig = visualize_samples(
                    st.session_state.x_test,
                    st.session_state.y_test,
                    predictions,
                    num_samples=5
                )
                st.pyplot(fig)
                plt.close()

            st.divider()

            if st.button("💾 Save Model"):
                st.session_state.model.save('mnist_cnn_model.keras')
                st.success("Model saved as 'mnist_cnn_model.keras'")
        else:
            st.info("👈 Train the model first to see results")
else:
    st.info("👈 Click 'Load Dataset' in the sidebar to begin")

st.divider()

with st.expander("ℹ️ About this App"):
    st.markdown("""
    This application demonstrates a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset.

    **Features:**
    - Interactive model training with real-time progress
    - Configurable hyperparameters (epochs, batch size)
    - Visualization of training metrics
    - Sample prediction display
    - Model persistence

    **Architecture:**
    - 2 Convolutional blocks with batch normalization and dropout
    - MaxPooling for spatial dimension reduction
    - Dense layers for classification
    - Optimized to achieve >95% test accuracy
    """)
