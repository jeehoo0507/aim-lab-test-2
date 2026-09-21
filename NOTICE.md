# Sources

`coco_kd/models.py` is a torch-only adaptation of the official MaskedKD DeiT implementation, carried forward from experiment 1. Original copyright notices are retained. The unmodified reference model/loss/README files and exact revision are in `vendor/MaskedKD/`.

Official source: https://github.com/effl-lab/MaskedKD

The new data preparation, training orchestration, logging and analysis implement a separate COCO single-label classification experiment. They do not reproduce the full ImageNet training protocol from the paper.

COCO images and annotations retain their original terms; this repository distributes preparation code and derived experiment manifests, not the image dataset.
