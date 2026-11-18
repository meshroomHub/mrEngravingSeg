import copy
import json
import os
import re
import tempfile


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_sfm(path, temporary_folder):
    """Load an SfMData file as a JSON-compatible dictionary."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("Input SfMData does not exist: '{}'".format(path))
    if os.path.splitext(path)[1].lower() in (".sfm", ".json"):
        with open(path, encoding="utf-8") as sfm_file:
            return json.load(sfm_file)

    try:
        from pyalicevision import sfmData, sfmDataIO
    except ImportError as exc:
        raise RuntimeError(
            "This SfMData format requires pyalicevision; export the input as .sfm or .json."
        ) from exc

    data = sfmData.SfMData()
    if not sfmDataIO.load(data, path, sfmDataIO.ALL):
        raise RuntimeError("Cannot load the SfMData: '{}'".format(path))

    os.makedirs(temporary_folder, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(suffix=".sfm", dir=temporary_folder)
    os.close(descriptor)
    try:
        if not sfmDataIO.save(data, temporary_path, sfmDataIO.ALL):
            raise RuntimeError("Cannot convert the SfMData to JSON: '{}'".format(path))
        with open(temporary_path, encoding="utf-8") as sfm_file:
            return json.load(sfm_file)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def save_sfm(sfm, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as sfm_file:
        json.dump(sfm, sfm_file, indent=4)


def resolve_view_image(view, sfm_path):
    view_id = view.get("viewId")
    image_path = view.get("path")
    if view_id is None:
        raise ValueError("Every SfMData view must define viewId.")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("SfMData view {} has no valid image path.".format(view_id))

    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(os.path.abspath(sfm_path)), image_path)
    image_path = os.path.abspath(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError("Image for SfMData view {} does not exist: {}".format(view_id, image_path))
    if not image_path.lower().endswith(IMAGE_EXTENSIONS):
        raise ValueError("Unsupported image format for SfMData view {}: {}".format(view_id, image_path))
    return image_path


def resolve_mask(mask_path, view_id):
    if not mask_path:
        return None
    if os.path.isdir(mask_path):
        candidates = [
            os.path.join(mask_path, "{}{}".format(view_id, extension))
            for extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError("No mask found for SfMData view {} in '{}'.".format(view_id, mask_path))
    if not os.path.isfile(mask_path):
        raise FileNotFoundError("Mask does not exist: '{}'".format(mask_path))
    return mask_path


def process_views(sfm, sfm_path, model, predictor, output_folder, mask_path=None, stride=256, patch_size=512,
                  normalization="standard", mask_threshold=127, logger=None):
    """Predict each view and return a copied SfMData referencing the generated maps."""
    views = sfm.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("Input SfMData contains no views.")
    if stride <= 0 or patch_size <= 0 or stride > patch_size:
        raise ValueError("Stride and Patch Size must be positive, and Stride cannot exceed Patch Size.")

    output_sfm = copy.deepcopy(sfm)
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    normalize = predictor.get_normalization_function(normalization)
    used_view_ids = set()

    for index, view in enumerate(output_sfm["views"], 1):
        view_id = str(view.get("viewId", ""))
        if not view_id or view_id in used_view_ids:
            raise ValueError("SfMData views must have unique, non-empty viewId values.")
        used_view_ids.add(view_id)

        image_path = resolve_view_image(view, sfm_path)
        view_mask = resolve_mask(mask_path, view_id)
        safe_view_id = re.sub(r"[^A-Za-z0-9_.-]", "_", view_id)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(
            output_folder,
            "{:06d}_{}_{}_result.png".format(index, safe_view_id, image_name),
        )
        if logger:
            logger.info("Predicting image: {}".format(image_path))
        predictor.process_image(
            image_path,
            model,
            mask_path=view_mask,
            stride=stride,
            patch_size=patch_size,
            fun_img=normalize,
            output_dir=output_folder,
            mask_threshold=mask_threshold,
            output_path=output_path,
        )
        view["path"] = output_path

    return output_sfm
