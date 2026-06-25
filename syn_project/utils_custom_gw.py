import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LRScheduler

from shimmer import ContrastiveLoss, GWLosses2Domains, GlobalWorkspace2Domains, LatentsDomainGroupT, LatentsDomainGroupsT, LossOutput, ModelModeT, RawDomainGroupsT, SelectionBase, combine_loss
from shimmer.utils import group_batch_size, group_device
from shimmer.modules.global_workspace import OneCycleSchedulerSentinel

from .utils_attention import MyAttentionGWLosses

class CustomSelection(SelectionBase):
    def forward(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: LatentsDomainGroupT
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the module.

        Args:
            domains (`LatentsDomainGroupT`): input unimodal latent representations
            encodings_pre_fusion (`LatentsDomainGroupT`): pre-fusion domain latent
                representation.

        Returns:
            `dict[str, torch.Tensor]`: whether the domain is selected for each input
            in the batch.
        """
        selection: dict[str, torch.Tensor] = {}
        bs = group_batch_size(domains)
        n =len(domains.keys())
        coef = torch.full((bs,), 1.0 / n, device=group_device(domains))
        for domain in domains:
            selection[domain] = coef.clone()
        return selection

class MyCustomGWLosses(GWLosses2Domains):
    def __init__(self, gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn, custom_weights=None, noise=None):
        super().__init__(gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn)
        self.custom_weights = custom_weights

    def step(
        self,
        raw_data: RawDomainGroupsT,
        domain_latents: LatentsDomainGroupsT,
        mode: ModelModeT,
    ) -> LossOutput:

        metrics: dict[str, torch.Tensor] = {}

        metrics.update(self.demi_cycle_loss(domain_latents, raw_data))
        metrics.update(self.cycle_loss(domain_latents, raw_data))
        metrics.update(self.translation_loss(domain_latents, raw_data))
        metrics.update(self.contrastive_loss(domain_latents))

        custom_weights = self.custom_weights
     
        if custom_weights is None or len(custom_weights.items()) == 0:
            return LossOutput(combine_loss(metrics, self.loss_coefs), metrics)

        weighted_losses = []
        for key, weight in custom_weights.items():
            if key in metrics:
                weighted_losses.append(metrics[key] * weight)

        custom_loss = torch.stack(weighted_losses).sum()

        return LossOutput(custom_loss, metrics)

def remove_action_from_data(data):
    new_data = {}
    for key, value in data.items():
        # Retirer 'action' de la frozenset-clé
        new_key = frozenset(k for k in key if k != 'action')
        # Retirer 'action' du dict de latents si présent
        new_value = {k: v for k, v in value.items() if k != 'action'}
        new_data[new_key] = new_value
    return new_data

class CombinedGWLosses(nn.Module):
    def __init__(self, attention_loss: MyAttentionGWLosses, custom_loss: MyCustomGWLosses, attention_weight=1.0, custom_weight=1.0):
        super().__init__()
        self.attention_loss = attention_loss
        self.custom_loss = custom_loss
        self.attention_weight = attention_weight
        self.custom_weight = custom_weight

    @property
    def attention(self):
        """Délègue à MyAttentionGWLosses pour configure_optimizers."""
        return self.attention_loss.attention
        
    def step(
        self,
        raw_data: "RawDomainGroupsT",
        domain_latents: "LatentsDomainGroupsT",
        mode: "ModelModeT",
    ) -> "LossOutput":
        out_attention = self.attention_loss.step(raw_data, domain_latents, mode)

        domain_latents_custom = remove_action_from_data(
    {k: v for k, v in domain_latents.items() if frozenset({'action'}) != k}
)
        raw_data_custom = remove_action_from_data(
    {k: v for k, v in raw_data.items() if frozenset({'action'}) != k}
)
        out_custom = self.custom_loss.step(raw_data_custom, domain_latents_custom, mode)

        combined_loss = (
            self.attention_weight * out_attention.loss +
            self.custom_weight * out_custom.loss
        )

        # Merge des métriques avec préfixes pour distinguer dans wandb
        metrics = {
            **{f"attention/{k}": v for k, v in out_attention.metrics.items()},
            **{f"custom/{k}": v for k, v in out_custom.metrics.items()},
        }

        return LossOutput(combined_loss, metrics)

class MyGlobalWorkspace(GlobalWorkspace2Domains):
    def __init__(
            self,
            domain_mods,
            gw_encoders,
            gw_decoders,
            workspace_dim,
            loss_coefs,
            custom_weights,
            noise,
            attention_tree_config=None,
            fusion_activation_function = torch.nn.Identity(),
            modules_to_freeze=[],
            attention_lr=1e-1,
            attention_weight_decay=0.0,
            *args,
            **kwargs):
        kwargs.pop('fusion_activation_fn', None)
        super().__init__(domain_mods, gw_encoders, gw_decoders, workspace_dim, loss_coefs, fusion_activation_fn=fusion_activation_function, *args, **kwargs)

        contrastive_loss = ContrastiveLoss(
                torch.tensor([1 / 0.07]).log(), "mean", False
            )
        
        self.modules_to_freeze=modules_to_freeze
        self.attention_lr = attention_lr
        self.attention_weight_decay = attention_weight_decay
        selection_mod = CustomSelection()
        attention_losses = MyAttentionGWLosses(
            self.gw_mod,
            selection_mod,
            self.domain_mods,
            loss_coefs,
            contrastive_loss,
            attention_tree_config
        )

        domain_mods_custom_loss = {k: self.domain_mods[k] for k in self.domain_mods.keys() - {'action'}}

        custom_losses = MyCustomGWLosses(
            self.gw_mod,
            selection_mod,
            domain_mods_custom_loss,
            loss_coefs,
            contrastive_loss,
            custom_weights=custom_weights,
            noise=noise
        )

        self.loss_mod = CombinedGWLosses(
            attention_losses,
            custom_losses,
            attention_weight=1.0,
            custom_weight=1.0
)
        
    def configure_optimizers(self):
            attention_params = list(self.loss_mod.attention.parameters())
            attention_param_ids = {id(p) for p in attention_params}

            other_params = [
                p for p in self.parameters()
                if id(p) not in attention_param_ids
            ]

            optimizer = AdamW(
                [
                    {"params": other_params, "lr": self.optim_lr, "weight_decay": self.optim_weight_decay},
                    {"params": attention_params, "lr": self.attention_lr, "weight_decay": self.attention_weight_decay},
                ],
            )

            if self.scheduler is None:
                return {"optimizer": optimizer}

            lr_scheduler: LRScheduler
            if isinstance(self.scheduler, OneCycleSchedulerSentinel):
                # max_lr doit matcher le nombre de groupes
                scheduler_args = dict(self.scheduler_args)
                base_max_lr = scheduler_args.pop("max_lr")
                scheduler_args["max_lr"] = [base_max_lr, base_max_lr]  # peu importe pour le groupe 1, écrasé après
                lr_scheduler = OneCycleLR(optimizer, **scheduler_args)
            else:
                lr_scheduler = self.scheduler(optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                },
            }

    def on_train_start(self):            
            for name in self.modules_to_freeze:
                if name in self.gw_mod.gw_encoders:
                    encoder = self.gw_mod.gw_encoders[name]
                    decoder = self.gw_mod.gw_decoders[name]
                    
                    for param in encoder.parameters():
                        param.requires_grad = False
                    
                    for param in decoder.parameters():
                        param.requires_grad = False

                    encoder.eval()
                    decoder.eval()
                    
                    print(f"❄️ Module {name} freeze avec succès.")

    def on_train_epoch_start(self):
            for name in self.modules_to_freeze:
                if name in self.gw_mod.gw_encoders:
                    self.gw_mod.gw_encoders[name].eval()
                    self.gw_mod.gw_decoders[name].eval()
