import torch
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
     
        if len(custom_weights.items()) == 0:
            return LossOutput(combine_loss(metrics, self.loss_coefs), metrics)

        weighted_losses = []
        for key, weight in custom_weights.items():
            if key in metrics:
                weighted_losses.append(metrics[key] * weight)

        custom_loss = torch.stack(weighted_losses).sum()

        return LossOutput(custom_loss, metrics)

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
        self.loss_mod = MyAttentionGWLosses(
            self.gw_mod, 
            selection_mod, 
            self.domain_mods, 
            loss_coefs,
            contrastive_loss,
        )

    def configure_optimizers(self):
            attention_params = list(self.loss_mod.attention.parameters())
            attention_param_ids = {id(p) for p in attention_params}

            other_params = [
                p for p in self.parameters()
                if id(p) not in attention_param_ids
            ]

            # un seul optimizer, deux groupes -- lr initial différent
            # (le lr du groupe attention sera de toute façon écrasé à chaque step
            # par le callback FixAttentionLR, donc sa valeur ici importe peu)
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
                    
                    # 1. Bloquer le calcul des gradients (Freeze des poids)
                    for param in encoder.parameters():
                        param.requires_grad = False
                    
                    for param in decoder.parameters():
                        param.requires_grad = False

                    # 2. Passer en mode évaluation (Désactive Dropout/BatchNorm)
                    encoder.eval()
                    decoder.eval()
                    
                    print(f"❄️ Module {name} freeze avec succès.")

    def on_train_epoch_start(self):
            for name in self.modules_to_freeze:
                if name in self.gw_mod.gw_encoders:
                    self.gw_mod.gw_encoders[name].eval()
                    self.gw_mod.gw_decoders[name].eval()
