from pathlib import Path

import numpy as np

from app.config import DEFAULT_MODEL_FILES


class WordRecognizer:
    def __init__(
        self,
        model_path: Path | None = None,
        labels_path: Path | None = None,
    ):
        if model_path is None or labels_path is None:
            default_model_path, default_labels_path = self._default_model_and_labels()
            self.model_path = Path(model_path or default_model_path)
            self.labels_path = Path(labels_path or default_labels_path)
        else:
            self.model_path = Path(model_path)
            self.labels_path = Path(labels_path)
        self.labels = self._load_labels()
        self.backend, self.model = self._load_model()
        if self.backend == "tflite":
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()
        else:
            self.input_details = None
            self.output_details = None

    def predict_batch(self, images_rgb: list[np.ndarray], k: int = 1) -> list[list[tuple[str, float]]]:
        return [self.predict_topk(image_rgb, k=k) for image_rgb in images_rgb]

    def predict_topk(self, image_rgb: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if self.backend == "keras":
            input_data = np.expand_dims(image_rgb.astype(np.float32), axis=0)
            output = self.model.predict(input_data, verbose=0)[0]
        else:
            input_detail = self.input_details[0]
            input_data = self._prepare_input(image_rgb, input_detail)
            self.model.set_tensor(input_detail["index"], input_data)
            self.model.invoke()
            output = self.model.get_tensor(self.output_details[0]["index"])[0]
            output = self._dequantize_output(output, self.output_details[0])

        output = np.asarray(output, dtype=np.float32)
        if k <= 1:
            top_idx = [int(np.argmax(output))]
        else:
            top_idx = output.argsort()[-k:][::-1]
        return [(self.labels[int(index)], float(output[int(index)]) * 100.0) for index in top_idx]

    @staticmethod
    def _default_model_and_labels() -> tuple[Path, Path]:
        for model_path, labels_path in DEFAULT_MODEL_FILES:
            if model_path.exists() and labels_path.exists():
                return model_path, labels_path
        return DEFAULT_MODEL_FILES[0]

    def _load_labels(self) -> list[str]:
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        with open(self.labels_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Recognition model not found: {self.model_path}")

        if self.model_path.suffix.lower() in {".keras", ".h5"}:
            try:
                import tensorflow as tf  # type: ignore

                return "keras", tf.keras.models.load_model(self.model_path)
            except Exception:
                try:
                    import keras  # type: ignore

                    return "keras", keras.models.load_model(self.model_path)
                except Exception as exc:
                    raise RuntimeError(
                        "Install TensorFlow/Keras to run the active .keras recognition model."
                    ) from exc

        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except Exception:
            try:
                import tensorflow as tf  # type: ignore

                Interpreter = tf.lite.Interpreter
            except Exception as exc:
                raise RuntimeError(
                    "Install tflite-runtime or tensorflow to run model inference."
                ) from exc

        interpreter = Interpreter(model_path=str(self.model_path))
        interpreter.allocate_tensors()
        return "tflite", interpreter

    @staticmethod
    def _prepare_input(image_rgb: np.ndarray, input_detail) -> np.ndarray:
        input_dtype = input_detail["dtype"]
        input_data = np.expand_dims(image_rgb, axis=0)

        if input_dtype == np.float32:
            return input_data.astype(np.float32)

        scale, zero_point = input_detail.get("quantization", (0.0, 0))
        if scale:
            input_data = input_data / scale + zero_point
        return input_data.astype(input_dtype)

    @staticmethod
    def _dequantize_output(output, output_detail) -> np.ndarray:
        output = output.astype(np.float32)
        scale, zero_point = output_detail.get("quantization", (0.0, 0))
        if scale:
            output = (output - zero_point) * scale
        return output
