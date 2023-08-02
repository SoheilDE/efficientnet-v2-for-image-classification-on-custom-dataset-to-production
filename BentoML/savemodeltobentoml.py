import bentoml
from pathlib import Path
import tensorflow as tf


def load_and_save_model(model_path):
    model = tf.keras.models.load_model(model_path)

    # Manually compile model
    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Create BentoService
    bento_service = bentoml.keras.save_model("efficientnet_model", model)
    print(f"BentoML model tag: {bento_service.tag}")


if __name__ == '__main__':
    # Model path
    model_path = Path("model_path")

    # Load and save model
    load_and_save_model(model_path)
