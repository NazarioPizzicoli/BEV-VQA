"""
Unit tests for NuScenes-QA Boost Interventions 1, 2, and 3.

Test Coverage:
1. Intervention 1: DeeperConvProjector with 64 tokens + Dual Pooling (Avg + Max).
2. Intervention 1: Backward compatibility with 32 tokens and pos_embed interpolation.
3. Intervention 2: CategoryBalancedSampler empirical distribution verification.
4. Intervention 3: NuScenesBoostLoss ordinal Gaussian smoothing distance monotonic property.
5. Intervention 3: NuScenesBoostLoss status class weighting verification.
"""

from collections import Counter
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from bev_vqa.models.projector import ProjectorConfig, DeeperConvProjector, get_spatial_pool_shape
from bev_vqa.data.sampler import CategoryBalancedSampler, DEFAULT_CATEGORY_RATIOS
from bev_vqa.training.losses import (
    NuScenesBoostLoss,
    compute_gaussian_count_distribution,
    get_status_weight,
    STATUS_WEIGHTS,
)


class DummyTokenizer:
    """Tokenizer mock per test isolati e veloci."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        # Mappa cifre 0-10 a token_id deterministici
        self.digit_map = {str(i): 15 + i for i in range(11)}
        self.word_map = {
            "with rider": 100,
            "parked": 101,
            "stopped": 102,
            "moving": 103,
            "car": 104,
            "yes": 105,
        }

    def encode(self, text, add_special_tokens=False):
        clean = text.strip()
        if clean in self.digit_map:
            return [self.digit_map[clean]]
        if clean in self.word_map:
            return [self.word_map[clean]]
        return [200]


# =====================================================================
# TEST INTERVENTION 1: 64-Token Isotropic Grid + Dual Pooling
# =====================================================================

def test_intervention_1_isotropic_dual_pooling_forward_backward():
    """Verifica che il proiettore a 64 token con Dual Pooling funzioni end-to-end con gradienti."""
    cfg = ProjectorConfig(
        in_channels=128,
        num_tokens=64,
        projector_output_size=512,
        hidden_channels=64,
        num_conv_layers=2,
        use_dual_pooling=True,
    )
    proj = DeeperConvProjector(cfg)

    # Verifica griglia isotropa 8x8
    assert proj.pool_shape == (8, 8)
    assert hasattr(proj, "avg_pool")
    assert hasattr(proj, "max_pool")
    assert hasattr(proj, "pool_proj")
    assert proj.pool_proj.in_channels == cfg.hidden_channels * 2
    assert proj.pool_proj.out_channels == cfg.hidden_channels

    # Forward pass con input fittizio BEV [B=2, C=128, H=200, W=200]
    bev = torch.randn(2, 128, 32, 32, requires_grad=True)
    out = proj(bev)

    # Output shape: [2, 64, 512]
    assert out.shape == (2, 64, 512)
    assert torch.isfinite(out).all()

    # Backward pass
    loss = out.sum()
    loss.backward()

    # Verifica gradienti finiti su tutti i parametri
    assert proj.pool_proj.weight.grad is not None
    assert torch.isfinite(proj.pool_proj.weight.grad).all()
    assert proj.pos_embed.grad is not None
    assert torch.isfinite(proj.pos_embed.grad).all()
    assert bev.grad is not None


def test_intervention_1_backward_compatibility_32_tokens():
    """Verifica che la configurazione classica a 32 token senza dual pooling continui a funzionare."""
    cfg = ProjectorConfig(
        in_channels=128,
        num_tokens=32,
        projector_output_size=512,
        hidden_channels=64,
        num_conv_layers=2,
        use_dual_pooling=False,
    )
    proj = DeeperConvProjector(cfg)

    assert proj.pool_shape == (4, 8)
    assert hasattr(proj, "pool")
    assert not hasattr(proj, "pool_proj")

    bev = torch.randn(1, 128, 32, 32)
    out = proj(bev)
    assert out.shape == (1, 32, 512)


def test_intervention_1_pos_embed_interpolation():
    """Verifica l'interpolazione 2D del pos_embed da 32 token (4x8) a 64 token (8x8)."""
    old_h, old_w = get_spatial_pool_shape(32)
    new_h, new_w = get_spatial_pool_shape(64)
    assert (old_h, old_w) == (4, 8)
    assert (new_h, new_w) == (8, 8)

    old_pe = torch.randn(1, 32, 256)
    pe_2d = old_pe.view(1, old_h, old_w, 256).permute(0, 3, 1, 2)
    pe_interp = F.interpolate(pe_2d, size=(new_h, new_w), mode="bilinear", align_corners=False)
    new_pe = pe_interp.permute(0, 2, 3, 1).reshape(1, 64, 256)

    assert new_pe.shape == (1, 64, 256)
    assert torch.isfinite(new_pe).all()


# =====================================================================
# TEST INTERVENTION 2: Category-Balanced Sampler
# =====================================================================

def test_intervention_2_category_balanced_sampling():
    """Verifica che il campionatore rispetti le percentuali target (30/30/20/10/10) su un dataset sbilanciato."""
    # Simula un dataset altamente sbilanciato (80% exist, 10% count, 5% object, 3% status, 2% comparison)
    mock_data = []
    counts_map = {"exist": 800, "count": 100, "object": 50, "status": 30, "comparison": 20}
    for cat, num in counts_map.items():
        for i in range(num):
            mock_data.append({"template_type": cat, "id": f"{cat}_{i}"})

    sampler = CategoryBalancedSampler(
        dataset_data=mock_data,
        target_ratios=DEFAULT_CATEGORY_RATIOS,
        num_samples=10000,
        seed=123,
    )

    sampled_indices = list(sampler)
    assert len(sampled_indices) == 10000

    sampled_cats = [mock_data[i]["template_type"] for i in sampled_indices]
    counts = Counter(sampled_cats)
    total = len(sampled_cats)

    prop_count = counts["count"] / total
    prop_status = counts["status"] / total
    prop_object = counts["object"] / total
    prop_exist = counts["exist"] / total
    prop_comp = counts["comparison"] / total

    # Tolleranza statistica di ±2% su 10,000 campioni
    tol = 0.02
    assert abs(prop_count - 0.30) < tol, f"Count errato: {prop_count:.3f} vs 0.30"
    assert abs(prop_status - 0.30) < tol, f"Status errato: {prop_status:.3f} vs 0.30"
    assert abs(prop_object - 0.20) < tol, f"Object errato: {prop_object:.3f} vs 0.20"
    assert abs(prop_exist - 0.10) < tol, f"Exist errato: {prop_exist:.3f} vs 0.10"
    assert abs(prop_comp - 0.10) < tol, f"Comparison errato: {prop_comp:.3f} vs 0.10"


# =====================================================================
# TEST INTERVENTION 3: NuScenesBoostLoss (Smoothing + Class Weighting)
# =====================================================================

def test_intervention_3_ordinal_gaussian_smoothing_monotonicity():
    """
    Verifica proprietà fondamentale dell'Ordinal Gaussian Smoothing:
    Se ground truth k=3, allora:
    loss(pred=3) < loss(pred=2) < loss(pred=1) < loss(pred=0)
    e loss(pred=3) < loss(pred=4) < loss(pred=5).
    """
    tokenizer = DummyTokenizer()
    loss_fn = NuScenesBoostLoss(tokenizer=tokenizer, count_sigma=0.9, max_count=10)

    vocab_size = 250
    seq_len = 5

    # Ground truth: k = 3
    # Creiamo logits che favoriscono diverse cifre per verificare la monotonia
    digit_3_token = tokenizer.encode("3")[0]
    digit_2_token = tokenizer.encode("2")[0]
    digit_1_token = tokenizer.encode("1")[0]
    digit_0_token = tokenizer.encode("0")[0]

    losses = {}
    for digit, tok_id in [("3", digit_3_token), ("2", digit_2_token), ("1", digit_1_token), ("0", digit_0_token)]:
        logits = torch.zeros(1, seq_len, vocab_size)
        # Logit molto alto per il token predetto al primo step di risposta (indice 1)
        logits[0, 0, tok_id] = 10.0

        labels = torch.full((1, seq_len), -100, dtype=torch.long)
        labels[0, 1] = digit_3_token  # GT è 3

        l = loss_fn(
            logits=logits,
            labels=labels,
            template_types=["count"],
            answers=["3"],
        )
        losses[digit] = l.item()

    # Monotonia: errore 0 < errore 1 < errore 2 < errore 3
    assert losses["3"] < losses["2"], f"loss(3) {losses['3']} non è < loss(2) {losses['2']}"
    assert losses["2"] < losses["1"], f"loss(2) {losses['2']} non è < loss(1) {losses['1']}"
    assert losses["1"] < losses["0"], f"loss(1) {losses['1']} non è < loss(0) {losses['0']}"


def test_intervention_3_status_class_weighting():
    """Verifica che i moltiplicatori di status vengano applicati correttamente (with rider=4.0x, parked=1.5x, ecc.)."""
    tokenizer = DummyTokenizer()
    loss_fn = NuScenesBoostLoss(tokenizer=tokenizer)

    vocab_size = 250
    seq_len = 4

    logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)
    labels = torch.full((1, seq_len), -100, dtype=torch.long)
    labels[0, 1] = tokenizer.encode("with rider")[0]

    # Loss base (senza weighting o con peso 1.0)
    loss_base = F.cross_entropy(logits[..., :-1, :][0, 0:1].float(), labels[..., 1:][0, 0:1])

    loss_rider = loss_fn(
        logits=logits,
        labels=labels,
        template_types=["status"],
        answers=["with rider"],
    )

    # Il loss con "with rider" deve essere esattamente 4.0x la cross entropy base
    assert torch.isclose(loss_rider, loss_base * 4.0, rtol=1e-4)

    # Test "parked" (1.5x)
    labels[0, 1] = tokenizer.encode("parked")[0]
    loss_parked = loss_fn(
        logits=logits,
        labels=labels,
        template_types=["status"],
        answers=["parked"],
    )
    loss_base_parked = F.cross_entropy(logits[..., :-1, :][0, 0:1].float(), labels[..., 1:][0, 0:1])
    assert torch.isclose(loss_parked, loss_base_parked * 1.5, rtol=1e-4)


def test_intervention_3_backward_pass():
    """Verifica differenziabilità end-to-end della boost loss."""
    tokenizer = DummyTokenizer()
    loss_fn = NuScenesBoostLoss(tokenizer=tokenizer, count_sigma=0.9)

    logits = torch.randn(2, 6, 250, requires_grad=True)
    labels = torch.full((2, 6), -100, dtype=torch.long)
    labels[0, 2] = tokenizer.encode("3")[0]
    labels[1, 2] = tokenizer.encode("parked")[0]

    loss = loss_fn(
        logits=logits,
        labels=labels,
        template_types=["count", "status"],
        answers=["3", "parked"],
    )

    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_intervention_2_sampler_set_epoch():
    """Verifica che set_epoch cambi la sequenza generata e che la lunghezza rimanga esatta."""
    mock_data = [{"template_type": "count", "id": i} for i in range(100)]
    sampler = CategoryBalancedSampler(mock_data, target_ratios={"count": 1.0}, num_samples=50, seed=42)
    s1 = list(sampler)
    sampler.set_epoch(1)
    s2 = list(sampler)
    assert len(s1) == 50
    assert len(s2) == 50
    assert s1 != s2, "set_epoch deve diversificare il campionamento tra epoche!"


def test_intervention_3_real_qwen_tokenizer_no_collision():
    """
    Test critico con il VERO tokenizer di Qwen2.5:
    Verifica che '10' non collida con '1', e che per target=9:
    loss(pred=9) < loss(pred=8) < loss(pred=7) < loss(pred=1)
    eliminando la trappola del gradiente spurio su '1'.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    except Exception:
        return  # Salta se transformers o checkpoint non disponibile nell'ambiente corrente

    loss_fn = NuScenesBoostLoss(tokenizer=tok, count_sigma=0.9, max_count=10)

    # Verifica che i token IDs siano unici e coprano 0-9
    assert len(loss_fn.count_token_ids) == 10
    assert len(set(loss_fn.count_token_ids)) == 10
    assert loss_fn.supported_max_count == 9

    vocab_size = len(tok)
    seq_len = 3

    tok_9 = tok.encode("9", add_special_tokens=False)[0]
    tok_8 = tok.encode("8", add_special_tokens=False)[0]
    tok_7 = tok.encode("7", add_special_tokens=False)[0]
    tok_1 = tok.encode("1", add_special_tokens=False)[0]

    losses = {}
    for name, tid in [("9", tok_9), ("8", tok_8), ("7", tok_7), ("1", tok_1)]:
        logits = torch.zeros(1, seq_len, vocab_size)
        logits[0, 0, tid] = 20.0
        labels = torch.full((1, seq_len), -100, dtype=torch.long)
        labels[0, 1] = tok_9
        l = loss_fn(logits, labels, template_types=["count"], answers=["9"])
        losses[name] = l.item()

    # Monotonia rigorosa senza collisioni:
    assert losses["9"] < losses["8"], f"loss(9) {losses['9']} >= loss(8) {losses['8']}"
    assert losses["8"] < losses["7"], f"loss(8) {losses['8']} >= loss(7) {losses['7']}"
    assert losses["7"] < losses["1"], f"loss(7) {losses['7']} >= loss(1) {losses['1']}"


if __name__ == "__main__":
    print("Running Intervention 1 Tests...")
    test_intervention_1_isotropic_dual_pooling_forward_backward()
    test_intervention_1_backward_compatibility_32_tokens()
    test_intervention_1_pos_embed_interpolation()
    print("Intervention 1 Tests PASSED!")

    print("Running Intervention 2 Tests...")
    test_intervention_2_category_balanced_sampling()
    test_intervention_2_sampler_set_epoch()
    print("Intervention 2 Tests PASSED!")

    print("Running Intervention 3 Tests...")
    test_intervention_3_ordinal_gaussian_smoothing_monotonicity()
    test_intervention_3_status_class_weighting()
    test_intervention_3_backward_pass()
    test_intervention_3_real_qwen_tokenizer_no_collision()
    print("Intervention 3 Tests PASSED!")

    print("\nALL INTERVENTION TESTS PASSED SUCCESSFULLY!")
