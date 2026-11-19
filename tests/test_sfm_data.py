import json
import os

import numpy as np
import pytest
from PIL import Image

from src import sfm_data


def test_process_views_updates_sfm_paths_and_preserves_input(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("front.png", "side.png"):
        Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(images / name)
    sfm_path = tmp_path / "input.sfm"
    sfm = {
        "views": [
            {"viewId": "10", "poseId": "1", "path": "images/front.png"},
            {"viewId": "11", "poseId": "2", "path": "images/side.png"},
        ],
        "poses": [{"poseId": "1"}, {"poseId": "2"}],
        "intrinsics": [{"intrinsicId": "1", "width": "4", "height": "3"}],
    }
    sfm_path.write_text(json.dumps(sfm), encoding="utf-8")
    output_folder = tmp_path / "predictions"

    class Predictor:
        @staticmethod
        def get_normalization_function(name):
            return lambda values: values

        @staticmethod
        def process_image(image_path, model, **kwargs):
            Image.fromarray(np.full((3, 4), 127, dtype=np.uint8)).save(kwargs["output_path"])

    result = sfm_data.process_views(sfm, str(sfm_path), None, Predictor, str(output_folder))

    assert [view["viewId"] for view in result["views"]] == ["10", "11"]
    assert [view["poseId"] for view in result["views"]] == ["1", "2"]
    assert result["poses"] == sfm["poses"]
    assert result["intrinsics"] == sfm["intrinsics"]
    assert all(os.path.isfile(view["path"]) for view in result["views"])
    assert [os.path.basename(view["path"]) for view in result["views"]] == ["1.png", "2.png"]
    assert sfm["views"][0]["path"] == "images/front.png"

    output_sfm_path = tmp_path / "out" / "result.sfm"
    sfm_data.save_sfm(result, str(output_sfm_path))
    saved = json.loads(output_sfm_path.read_text(encoding="utf-8"))
    assert [view["path"] for view in saved["views"]] == [view["path"] for view in result["views"]]


def test_process_views_rejects_repeated_pose_ids(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("front.png", "side.png"):
        Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(images / name)
    sfm = {
        "views": [
            {"viewId": "10", "poseId": "1", "path": str(images / "front.png")},
            {"viewId": "11", "poseId": "1", "path": str(images / "side.png")},
        ],
    }

    class Predictor:
        @staticmethod
        def get_normalization_function(name):
            return lambda values: values

        @staticmethod
        def process_image(image_path, model, **kwargs):
            Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(kwargs["output_path"])

    with pytest.raises(ValueError, match="unique.*poseId"):
        sfm_data.process_views(sfm, str(tmp_path / "input.sfm"), None, Predictor, str(tmp_path / "out"))


def test_resolve_mask_by_view_id(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    mask_path = masks / "42.png"
    mask_path.touch()

    assert sfm_data.resolve_mask(str(masks), "42") == str(mask_path)


def test_resolve_mask_by_source_image_filename(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    mask_path = masks / "00000000.png"
    mask_path.touch()

    assert sfm_data.resolve_mask(
        str(masks),
        "1258694240",
        "/images/00000000.jpg",
    ) == str(mask_path)


def test_process_views_passes_mask_named_like_source_image(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    image_path = images / "00000000.png"
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(image_path)
    masks = tmp_path / "masks"
    masks.mkdir()
    mask_path = masks / "00000000.png"
    Image.fromarray(np.full((3, 4), 255, dtype=np.uint8)).save(mask_path)
    sfm_path = tmp_path / "input.sfm"
    sfm = {"views": [{"viewId": "1258694240", "poseId": "1258694240", "path": str(image_path)}]}

    class Predictor:
        @staticmethod
        def get_normalization_function(name):
            return lambda values: values

        @staticmethod
        def process_image(image_path, model, **kwargs):
            assert kwargs["mask_path"] == str(mask_path)
            Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(kwargs["output_path"])

    sfm_data.process_views(sfm, str(sfm_path), None, Predictor, str(tmp_path / "out"), str(masks))


def test_missing_view_mask_is_reported(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()

    with pytest.raises(FileNotFoundError, match="view 42"):
        sfm_data.resolve_mask(str(masks), "42")
