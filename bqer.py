# bqer.py
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Dict, List


# --------------------- Layer definition --------------------------------------

class BQERDecoderLayer(nn.Module):
    """
    TCEC-like residual correction.
    """
    def __init__(
        self,
        inner_layer: nn.Module,
        hidden_size: int,
        K_current: Optional[torch.Tensor] = None,   # (G,)
        K_prev: Optional[torch.Tensor] = None,      # unused (m=0)
        window_m: int = 0,
        alpha: float = 0.75,
        place: str = "post",
        group_size: int = 1,                        # >=1; 1 == channel-wise
    ):
        super().__init__()
        assert place in ("post",)
        assert window_m in (0, 1)
        assert hidden_size % group_size == 0, f"hidden_size {hidden_size} must be divisible by group_size {group_size}"

        self.inner = inner_layer
        self.hidden_size = hidden_size
        self.window_m = int(window_m)
        self.alpha = float(alpha)
        self.place = place
        self.group_size = int(group_size)

        C = hidden_size
        G = (C + self.group_size - 1) // self.group_size
        Kg = torch.zeros(G, dtype=torch.float32)
        if K_current is not None:
            assert K_current.dim() == 1 and K_current.numel() == G, f"Expected K_current len={G}"
            Kg = K_current.detach().to(torch.float32)
        self.register_buffer("K_groups", Kg, persistent=True)  # (G,)
        self.register_buffer("_prev_y", None, persistent=False)

    @torch.no_grad()
    def set_K(self, K_current: torch.Tensor, K_prev: Optional[torch.Tensor] = None):
        C = self.hidden_size
        G = (C + self.group_size - 1) // self.group_size
        assert K_current.dim() == 1 and K_current.numel() == G
        self.K_groups = K_current.detach().to(self.K_groups.device, dtype=torch.float32)

    def reset_cache(self):
        self._prev_y = None

    def _expanded_K(self, like: torch.Tensor) -> torch.Tensor:
        C = self.hidden_size
        k = torch.repeat_interleave(self.K_groups, self.group_size)[:C]  # (C,)
        return k.view(1, 1, -1).to(device=like.device, dtype=like.dtype)

    @torch.no_grad()
    def _compute_delta(self, y_cur: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0.0:
            return torch.zeros_like(y_cur)
        K = self._expanded_K(y_cur)
        return self.alpha * (K * y_cur)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        x_in = hidden_states
        out = self.inner(hidden_states, *args, **kwargs)
        x_out, tail = (out, ()) if isinstance(out, torch.Tensor) else (out[0], out[1:])
        y_cur = x_out - x_in
        x_out = x_out + self._compute_delta(y_cur)
        self._prev_y = y_cur.detach()
        return x_out if isinstance(out, torch.Tensor) else (x_out, *tail)

# --------------------- Accumulate stats and calibrate Ks --------------------------------------

@torch.no_grad()
def accumulate_stats(fp_model: nn.Module,
                                 q_model: nn.Module,
                                 dataloader,
                                 device: str = "cuda",
                                 group_size: int = 1):
    # resolve L, C and layer lists
    if hasattr(q_model, "model") and hasattr(q_model.model, "layers"):
        L = len(q_model.model.layers)
        C = q_model.config.hidden_size
        q_layers = q_model.model.layers
        fp_layers = fp_model.model.layers
    else:
        L = len(q_model.layers)
        C = q_model.config.hidden_size
        q_layers = q_model.layers
        fp_layers = fp_model.layers

    # per-layer CPU accumulators (fp32)
    yq2  = {l: torch.zeros(C, dtype=torch.float32) for l in range(L)}
    yqyf = {l: torch.zeros(C, dtype=torch.float32) for l in range(L)}

    q_model.eval().to(device)
    fp_model.eval().to(device)

    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Streaming stats"):
        batch = {k: v.to(device) for k, v in batch.items()}

        # ---- pass 1: Q model -> yq2 and cache yq (compact bf16) ----
        yq_bt = [None] * L
        q_handles = []
        def q_hook_factory(i):
            def hook(mod, inputs, outputs):
                x_in  = inputs[0]
                x_out = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                yq = (x_out - x_in).reshape(-1, C)                    # (B*T, C) model dtype
                yq32 = yq.to(torch.float32)
                yq2[i].add_((yq32 * yq32).sum(dim=0).to("cpu"))       # Σ y_q^2
                yq_bt[i] = yq.to(torch.bfloat16)                      # compact GPU cache
                return outputs
            return hook
        for i, layer in enumerate(q_layers):
            q_handles.append(layer.register_forward_hook(q_hook_factory(i)))
        _ = q_model(**batch)
        for h in q_handles: h.remove()

        # ---- pass 2: FP model -> yqyf using cached yq ----
        fp_handles = []
        def fp_hook_factory(i):
            def hook(mod, inputs, outputs):
                x_in  = inputs[0]
                x_out = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                yf32 = (x_out - x_in).reshape(-1, C).to(torch.float32)  # (B*T,C)
                if yq_bt[i] is not None:
                    yq32 = yq_bt[i].to(torch.float32)
                    yqyf[i].add_((yq32 * yf32).sum(dim=0).to("cpu"))     # Σ y_q y_fp
                return outputs
            return hook
        for i, layer in enumerate(fp_layers):
            fp_handles.append(layer.register_forward_hook(fp_hook_factory(i)))
        _ = fp_model(**batch)
        for h in fp_handles: h.remove()

        del yq_bt
        torch.cuda.empty_cache()

    return yq2, yqyf


@torch.no_grad()
def calibrate_bqer(fp_model: nn.Module,
                   q_model: nn.Module,
                   dataloader,
                   device: str = "cuda",
                   lambda1: float = 1e-6,
                   clip: float = 0.5,
                   group_size: int = 1):
    assert group_size >= 1
    yq2, yqyf = accumulate_stats(fp_model, q_model, dataloader, device=device, group_size=group_size)

    if hasattr(q_model, "model") and hasattr(q_model.model, "layers"):
        L = len(q_model.model.layers); C = q_model.config.hidden_size
    else:
        L = len(q_model.layers);       C = q_model.config.hidden_size

    g = int(group_size)
    G = (C + g - 1) // g

    Ks_cur, Ks_prev = {}, {}
    for l in range(L):
        sqq = yq2[l]   # (C,) on CPU
        sfq = yqyf[l]  # (C,) on CPU

        pad = G * g - C
        if pad:
            sqq = torch.nn.functional.pad(sqq, (0, pad))
            sfq = torch.nn.functional.pad(sfq, (0, pad))

        sqq_g = sqq.view(G, g).sum(dim=1)    # (G,)
        sfq_g = sfq.view(G, g).sum(dim=1)    # (G,)

        K = (sfq_g - sqq_g) / (sqq_g + lambda1)  # (G,)
        if clip is not None:
            K = K.clamp(-clip, clip)

        Ks_cur[l]  = K.to(torch.float32)
        Ks_prev[l] = K.clone()
    return Ks_cur, Ks_prev


# --------------------- Apply BQER to model --------------------------------------

def wrap_model_with_bqer(model: nn.Module,
                         Ks_current: Dict[int, torch.Tensor],  # (G,)
                         Ks_prev: Optional[Dict[int, torch.Tensor]] = None,
                         window_m: int = 0,
                         alpha: float = 0.75,
                         group_size: int = 1):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers; hidden_size = model.config.hidden_size
    else:
        layers = model.layers;       hidden_size = model.config.hidden_size

    for idx, layer in enumerate(layers):
        Kc = Ks_current.get(idx)
        wrapper = BQERDecoderLayer(
            inner_layer=layer,
            hidden_size=hidden_size,
            K_current=Kc,        # expected length G
            window_m=window_m,
            alpha=alpha,
            place="post",
            group_size=group_size,
        )
        layers[idx] = wrapper


def apply_bqer(
    q_model: nn.Module,
    Ks_cur: Dict[int, torch.Tensor],
    Ks_prev: Dict[int, torch.Tensor],
    window_m: int = 0,
    alpha: float = 0.75,
    group_size: int = 1,
):
    wrap_model_with_bqer(
        q_model,
        Ks_current=Ks_cur,
        Ks_prev=Ks_prev,
        window_m=window_m,
        alpha=alpha,
        group_size=group_size,
    )
    for layer in (q_model.model.layers if hasattr(q_model, "model") else q_model.layers):
        if isinstance(layer, BQERDecoderLayer):
            layer.reset_cache()

