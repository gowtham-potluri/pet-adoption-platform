import os
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras import layers, models

def train_model():
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 4 
    EPOCHS = 3

    # Force CPU to prevent Metal/GPU hangs
    tf.config.set_visible_devices([], 'GPU')

    with mlflow.start_run():
        # 1. Modern Data Loading (Replaces ImageDataGenerator)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            'data/train',
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            'data/train',
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )

        # 2. Simple CNN Model
        model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Rescaling(1./255), # Rescaling moved inside the model
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])

        # 3. Train
        print("Starting training...")
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
        
        # 4. Save
        os.makedirs("models", exist_ok=True)
        model.save("models/baseline_model.h5")
        mlflow.log_artifact("models/baseline_model.h5")
        print("Success! Model saved to models/baseline_model.h5")

if __name__ == "__main__":
    train_model()