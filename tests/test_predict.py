import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow")

from src import predict_image


class ConstantModel:
    def predict(self, patch, verbose=0):
        return np.ones((1, patch.shape[1], patch.shape[2], 1), dtype=np.float32)


def test_mask_keeps_patches_that_partially_overlap(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8)).save(image_path)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 255
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
    assert np.all(result[:2, :2] == 255)
    assert np.all(result[2:, :] == 0)
    assert np.all(result[:, 2:] == 0)


def test_stride_cannot_exceed_patch_size(tmp_path):
    with pytest.raises(ValueError, match="Stride must not exceed Patch Size"):
        predict_image.process_image(
            str(tmp_path / "unused.png"),
            ConstantModel(),
            stride=3,
            patch_size=2,
        )
