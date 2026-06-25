from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback

from shimmer import GWLosses2Domains, LatentsDomainGroupT, LatentsDomainGroupsT, LossOutput, ModelModeT, RawDomainGroupsT, SelectionBase

"""
Généralisation de MyAttentionGWLosses pour une arborescence à 2 niveaux
configurable dynamiquement (au lieu d'avoir attr/color/color2 codés en dur).

Structure de l'arbre :

    root_domain  (ex: "v_latents")
        │  encode -> g
        │
        ├── decode(g) -> x_root
        │       │
        │       ├── encode(x_root)["attr"]  -> g_attr      (niveau 1, feuille -> on
        │       │                                           inclut g_attr dans z)
        │       │
        │       └── encode(x_root)["color"] -> g_color     (niveau 1, feuille -> on
        │                                                   inclut g_color dans z)
        │
        └── pour certains domaines de niveau 1 (ex: "attr"):
                decode(g_attr) -> x_attr
                    └── encode(x_attr)["color"] -> g_color2  (niveau 2, feuille)

Chaque feuille (qu'elle soit de niveau 1 ou 2) reçoit un nom de score unique
("attr", "color", "color2", ...) et est pondérée par le module d'attention,
puis sommée pour former z.
"""

class FixAttentionLR(Callback):
    """
    Après chaque step, écrase le lr du groupe 'attention' pour qu'il
    reste fixe, peu importe ce que fait OneCycleLR sur les autres groupes.
    """
    def __init__(self, attention_lr: float, attention_group_idx: int = 1):
        super().__init__()
        self.attention_lr = attention_lr
        self.attention_group_idx = attention_group_idx

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        optimizer = trainer.optimizers[0]
        optimizer.param_groups[self.attention_group_idx]["lr"] = self.attention_lr


class SimpleWeightSelection(SelectionBase):
    def __init__(
        self,
        domain_names=("attr", "color"),
        grad_multiplier=10000.0,
        init_weights: dict[str, float] | None = None,  # ex: {"attr": 2.0, "color": 0.5, "v_latents": 1.0}
    ):
        super().__init__()
        self.domain_names = list(domain_names)
        
        if init_weights is not None:
            # Initialisation personnalisée par domaine
            proportions = torch.tensor(
                [init_weights.get(name, 0.0) for name in self.domain_names],
                dtype=torch.float32
            )
            init_tensor = torch.log(proportions)
        else:
            # Comportement par défaut : tous à 0 → softmax uniforme
            init_tensor = torch.zeros(len(self.domain_names))
        
        self.raw_weights = nn.Parameter(init_tensor)
        self.raw_weights.register_hook(lambda grad: grad * grad_multiplier)

    def forward(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: LatentsDomainGroupT
    ) -> dict[str, torch.Tensor]:
        bs = next(iter(domains.values())).size(0)
        device = next(iter(domains.values())).device
        weights = torch.softmax(self.raw_weights, dim=0)
        selection = {}
        for i, name in enumerate(self.domain_names):
            selection[name] = weights[i].expand(bs).to(device)
        return selection


class IndependentWeight(nn.Module):
    """
    Poids scalaire appris, indépendant de tout softmax partagé.
    Passé par un sigmoid pour rester borné entre 0 et 1, mais pas contraint
    à sommer à 1 avec d'autres scores (contrairement à SimpleWeightSelection).
    """

    def __init__(self, init_value: float = 0.0, grad_multiplier: float = 10000.0):
        super().__init__()
        # init_value est le logit de départ (0.0 -> sigmoid(0) = 0.5 au départ)
        self.raw_weight = nn.Parameter(torch.tensor(float(init_value)))
        self.raw_weight.register_hook(lambda grad: grad * grad_multiplier)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        weight = torch.sigmoid(self.raw_weight)  # scalaire dans (0, 1)
        return weight.expand(batch_size).to(device)


@dataclass
class Level2Spec:
    """Un domaine de niveau 2, recalculé en repassant par decode/encode."""

    # domaine source de niveau 1 dont on repart (ex: "attr")
    from_domain: str
    # domaine qu'on récupère après avoir ré-encodé x_from_domain (ex: "color")
    target_domain: str
    # nom du score / clé dans z (par défaut: target_domain + "2")
    score_name: Optional[str] = None

    def get_score_name(self) -> str:
        return self.score_name or f"{self.target_domain}2"


@dataclass
class AttentionTreeConfig:
    """Décrit toute l'arborescence à 2 niveaux."""

    # domaine d'entrée utilisé pour construire g = encode(domain_latents)[root_input_domain]
    root_input_domain: str = "v_latents"

    level0_domains: List[str] = field(default_factory=list)

    # domaines de niveau 1 à extraire de x = decode(g)
    # (le nom du domaine sert aussi de score_name, ex: ["attr", "color"])
    level1_domains: List[str] = field(default_factory=list)

    # domaines de niveau 2 : chacun part d'un domaine de niveau 1, le décode,
    # ré-encode, et récupère un domaine cible
    level2_domains: List[Level2Spec] = field(default_factory=list)

    def all_score_names(self) -> List[str]:
        names = list(self.level0_domains)
        names += list(self.level1_domains)
        names += [spec.get_score_name() for spec in self.level2_domains]
        return names


class MyAttentionGWLosses(GWLosses2Domains):
    def __init__(
        self,
        gw_mod,
        selection_mod,
        domain_mods,
        loss_coefs,
        contrastive_fn,
        tree_config: Optional[AttentionTreeConfig] = None,
    ):
        super().__init__(gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn)

        self.tree_config = tree_config or AttentionTreeConfig()
        
        score_names = self.tree_config.all_score_names()

        self.attention = SimpleWeightSelection(
            domain_names=tuple(score_names),
            init_weights=None)

    def _compute_leaves(
        self, domain_latents: "LatentsDomainGroupsT", key: frozenset
    ) -> Dict[str, torch.Tensor]:
        """Calcule g pour chaque feuille de l'arbre (niveau 1 + niveau 2)."""
        cfg = self.tree_config

        g = self.gw_mod.encode(domain_latents[key])[cfg.root_input_domain]
        x = self.gw_mod.decode(g)
        g1 = self.gw_mod.encode(x)

        leaves: Dict[str, torch.Tensor] = {}
        g_level0: Dict[str, torch.Tensor] = {}
        g_level1: Dict[str, torch.Tensor] = {}

        for dom in cfg.level0_domains:
            g_dom = self.gw_mod.encode(domain_latents[key])[dom]
            g_level0[dom] = g_dom
            leaves[dom] = g_dom

        for dom in cfg.level1_domains:
            g_dom = g1[dom]
            g_level1[dom] = g_dom
            leaves[dom] = g_dom

        for spec in cfg.level2_domains:
            x_from = self.gw_mod.decode(g1[spec.from_domain])
            g_target = self.gw_mod.encode(x_from)[spec.target_domain]
            leaves[spec.get_score_name()] = g_target

        return leaves

    def step(
        self,
        raw_data: "RawDomainGroupsT",
        domain_latents: "LatentsDomainGroupsT",
        mode: "ModelModeT",
    ) -> "LossOutput":
        key = next(k for k in domain_latents.keys() if len(k) > 1)
        cats = torch.argmax(raw_data[key]["attr"][0], dim=1)
        target = cats.long()

        leaves = self._compute_leaves(domain_latents, key)

        if len(leaves) == 0:
            return LossOutput(0)
        
        scores = self.attention(leaves, leaves)
        z = sum(scores[name].unsqueeze(-1) * leaves[name] for name in leaves)

        pred = self.gw_mod.decode(z)["action"]
        custom_loss = F.cross_entropy(pred, target, reduction="mean")

        return LossOutput(
            custom_loss,
            {
                "cross_entropy": custom_loss.detach(),
                **{f"weight_{name}": scores[name].mean() for name in leaves},
            },
        )