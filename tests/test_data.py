import json

import pytest

from coco_kd.data import CocoSubset, validate_manifest
from coco_kd.prepare import strict_candidates, deduplicate
from coco_kd.synthetic import synthetic_data


def test_data_split_integrity_and_tampering(tmp_path):
    synthetic_data(tmp_path)
    m = validate_manifest(tmp_path)
    probe = CocoSubset(tmp_path, "probe")
    val = CocoSubset(tmp_path, "val")
    assert set(r["id"] for r in probe.records) <= set(r["id"] for r in val.records)
    row = probe[0]
    assert row["image"].shape == (3, 224, 224)
    assert row["foreground"].shape == (196,)
    m["images"][0]["probe"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Probe"):
        validate_manifest(tmp_path)


def test_filter_checks_all_annotations_not_just_selected_categories():
    info = {"categories": [{"id": 1, "name": "bird"}, {"id": 2, "name": "person"}],
            "images": [{"id": n} for n in (1, 2, 3)], "annotations": []}
    for image, category, crowd in [(1, 1, 0), (2, 1, 0), (2, 2, 0), (3, 1, 1)]:
        info["annotations"].append({"image_id": image, "category_id": category, "area": 10,
                                    "iscrowd": crowd, "segmentation": [[0, 0, 2, 0, 2, 2]]})
    result = list(strict_candidates(info))
    assert [r[0]["id"] for r in result] == [1]


def test_duplicate_priority_prevents_leakage():
    records = [{"id": 1, "source_split": "train2017", "pixel_sha256": "a", "dhash": 0},
               {"id": 2, "source_split": "val2017", "pixel_sha256": "a", "dhash": 0},
               {"id": 3, "source_split": "train2017", "pixel_sha256": "b", "dhash": 15}]
    kept, rejected = deduplicate(records)
    assert [r["id"] for r in kept] == [2]
    assert len(rejected) == 2


def test_segmentation_decodes_union_of_instances():
    from coco_kd.prepare import segmentation
    anns = [{"segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]]},
            {"segmentation": [[6, 6, 8, 6, 8, 8, 6, 8]]}]
    mask = segmentation(anns, 10, 10)
    assert mask.shape == (10, 10) and mask.sum() == 20
