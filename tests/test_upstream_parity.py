"""Execute untouched upstream class definitions without its legacy timm imports.

This compares logits, attention, masked teacher behavior, and student gradients
against the vendored official implementation, not another copy of our code.
"""
import ast
from functools import partial
from pathlib import Path

import torch
from torch import nn

from coco_kd.models import DeiT, DropPath


def official_class(filename):
    path = Path(__file__).resolve().parents[1] / "vendor" / "MaskedKD" / filename
    tree = ast.parse(path.read_text())
    keep = {"Mlp", "Attention", "Block", "PatchEmbed", "VisionTransformer"}
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in keep]
    namespace = {"torch": torch, "nn": nn, "partial": partial, "DropPath": DropPath,
                 "to_2tuple": lambda value: value if isinstance(value, tuple) else (value, value),
                 "trunc_normal_": nn.init.trunc_normal_}
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 - trusted, vendored upstream classes
    return namespace["VisionTransformer"]


def pair(filename):
    torch.manual_seed(12)
    ours = DeiT(24, 2, 3, drop_path=0)
    original = official_class(filename)(embed_dim=24, depth=2, num_heads=3, num_classes=2,
                                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    original.load_state_dict(ours.state_dict(), strict=True)
    return ours.eval(), original.eval()


def test_student_logits_attention_and_gradients_match_official():
    ours, original = pair("models_student.py")
    x = torch.randn(2, 3, 224, 224)
    logits, attention = ours(x, return_attention=True)
    reference_logits, reference_attention = original(x)
    torch.testing.assert_close(logits, reference_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(attention, reference_attention.mean(1)[:, 0, 1:], rtol=1e-5, atol=1e-6)
    assert torch.equal(attention.topk(98, 1).indices, reference_attention.mean(1)[:, 0, 1:].topk(98, 1).indices)
    logits.square().sum().backward()
    reference_logits.square().sum().backward()
    for (_, current), (_, reference) in zip(ours.named_parameters(), original.named_parameters()):
        torch.testing.assert_close(current.grad, reference.grad, rtol=1e-5, atol=1e-6)


def test_teacher_full_and_masked_match_official():
    ours, original = pair("models_teacher.py")
    x = torch.randn(2, 3, 224, 224)
    indices = torch.rand(2, 196).topk(98, 1).indices
    with torch.no_grad():
        torch.testing.assert_close(ours(x), original(x, indices, False), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(ours(x, indices), original(x, indices, True), rtol=1e-5, atol=1e-6)
        all_indices = torch.arange(196).expand(2, -1)
        torch.testing.assert_close(ours(x, all_indices), ours(x), rtol=1e-5, atol=1e-6)


def test_real_deit_model_dimensions():
    from coco_kd.config import Config
    from coco_kd.models import build_model
    for role, dim, heads in [("student", 192, 3), ("teacher", 384, 6)]:
        model = build_model(role, Config(), pretrained=False).eval()
        assert len(model.blocks) == 12
        assert model.pos_embed.shape == (1, 197, dim)
        assert model.blocks[0].attn.num_heads == heads
        with torch.no_grad():
            logits, attention = model(torch.randn(1, 3, 224, 224), return_attention=True)
        assert logits.shape == (1, 10) and attention.shape == (1, 196)
