import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow")

from src import predict_image


class ConstantModel:
    def predict(self, patch, verbose=0):
        return np.ones((1, patch.shape[1], patch.shape[2], 1), dtype=np.float32)


class ProbabilityGridModel:
    def predict(self, patch, verbose=0):
        return np.array([[[[0.2], [0.5]], [[0.8], [1.0]]]], dtype=np.float32)


def test_configure_tensorflow_device_logs_cpu_fallback(monkeypatch, capsys):
    monkeypatch.setattr(predict_image.tf.config, "list_physical_devices", lambda device_type: [])

    assert predict_image.configure_tensorflow_device() == []
    assert "inference will run on CPU" in capsys.readouterr().out


def test_configure_tensorflow_device_enables_gpu_memory_growth(monkeypatch):
    gpu = type("GPU", (), {"name": "/physical_device:GPU:0"})()
    configured_gpus = []
    monkeypatch.setattr(predict_image.tf.config, "list_physical_devices", lambda device_type: [gpu])
    monkeypatch.setattr(
        predict_image.tf.config.experimental,
        "set_memory_growth",
        lambda device, enabled: configured_gpus.append((device, enabled)),
    )

    assert predict_image.configure_tensorflow_device() == [gpu]
    assert configured_gpus == [(gpu, True)]


@pytest.mark.parametrize("mask_value", [1, 255])
def test_mask_is_applied_to_prediction_pixels(tmp_path, mask_value):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8)).save(image_path)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = mask_value
    Image.fromarray(mask).save(mask_path)

    output_path = predict_image.process_image(
        str(image_path),
        ConstantModel(),
        mask_path=str(mask_path),
        stride=2,
        patch_size=2,
        fun_img=lambda values: values,
        output_dir=str(tmp_path / "output"),
    )

    result = np.asarray(Image.open(output_path))
    expected = np.zeros((4, 4), dtype=np.uint8)
    expected[0, 0] = 255
    np.testing.assert_array_equal(result, expected)


def test_process_image_writes_thresholded_binary_map(tmp_path):
    image_path = tmp_path / "image.png"
    grayscale_path = tmp_path / "prediction.png"
    binary_path = tmp_path / "binary" / "prediction.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_path)

    output_path = predict_image.process_image(
        str(image_path),
        ProbabilityGridModel(),
        stride=2,
        patch_size=2,
        fun_img=lambda values: values,
        output_dir=str(tmp_path / "output"),
        output_path=str(grayscale_path),
        binary_output_path=str(binary_path),
        binary_threshold=127,
    )

    assert output_path == str(grayscale_path)
    np.testing.assert_array_equal(
        np.asarray(Image.open(grayscale_path)),
        np.array([[51, 127], [204, 255]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(Image.open(binary_path)),
        np.array([[0, 0], [255, 255]], dtype=np.uint8),
    )


def test_stride_cannot_exceed_patch_size(tmp_path):
    with pytest.raises(ValueError, match="Stride must not exceed Patch Size"):
        predict_image.process_image(
            str(tmp_path / "unused.png"),
            ConstantModel(),
            stride=3,
            patch_size=2,
        )
