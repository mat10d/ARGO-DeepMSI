import torch

from argo_deepmsi.models import wagner


def test_frozen_equivalence_and_attention_capture():
    torch.manual_seed(0)
    model = wagner.build_slide_transformer_for_test()  # small config helper
    x = torch.randn(1, 32, model.tile_dim)             # 32 tiles
    with torch.no_grad():
        p_off = model(x)
    with wagner.capture_attention(model), torch.no_grad():
        p_on = model(x)
    assert torch.allclose(p_off, p_on, atol=1e-6)      # hook must not perturb numerics
    attn = model.last_cls_attn
    assert attn.shape[-1] == 32
    assert abs(float(attn.sum()) - 1.0) < 1e-4
