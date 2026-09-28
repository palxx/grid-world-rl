"""Restricted action readout: hidden state -> logits over the 4 action tokens (no generation, no vocab softmax)."""
from __future__ import annotations

import torch


def unwrap(model):
    """CausalLM under an optional PEFT wrapper (LoRA layers stay active)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def backbone_and_head(model):
    base = unwrap(model)
    return base.get_decoder(), base.get_output_embeddings()


def action_logits_from_hidden(model, hidden: torch.Tensor, action_ids) -> torch.Tensor:
    """hidden [..., D] (post final norm) -> [..., 4] float32 logits over the action tokens."""
    _, head = backbone_and_head(model)
    W = head.weight[torch.as_tensor(action_ids, device=head.weight.device)]
    return hidden.float() @ W.float().T


def action_logits(model, input_ids, attention_mask, positions, action_ids):
    """Logits at `positions` [B, P] of a padded batch; only lm_head rows of the 4 actions are used."""
    dec, _ = backbone_and_head(model)
    h = dec(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    idx = positions.clamp(min=0)
    hp = h[torch.arange(h.shape[0], device=h.device)[:, None], idx]
    return action_logits_from_hidden(model, hp, action_ids)
