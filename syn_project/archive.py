def get_mask_from_shapes_diff(images: torch.Tensor, temperature: float = 10.0) -> torch.Tensor:
    """
    Version différentiable de l'extraction de masque.
    
    Args:
        images (torch.Tensor): Tensor de forme (B, C, H, W)
        temperature (float): Contrôle la "dureté" du seuil. 
                             Plus c'est élevé, plus c'est proche d'un masque binaire.
    
    Returns:
        torch.Tensor: Masque "soft" entre 0 et 1.
    """
    # 1. Conversion en niveaux de gris si nécessaire (moyenne des canaux RGB)
    if images.shape[1] == 3:
        grayscale = images.mean(dim=1, keepdim=True).clone()
    else:
        grayscale = images.clone()

    # 2. Normalisation optionnelle (si vos images ne sont pas déjà en [0, 1])
    # On suppose ici que le fond est à 0 et l'objet est > 0.
    
    # 3. Sigmoid "Hard" : Approximation de (x > 0)
    # On centre la sigmoid légèrement au dessus de 0 (ex: 0.05) pour éviter le bruit de fond
    threshold = 0.1
    masks = torch.sigmoid((grayscale - threshold) * temperature)
    
    return masks

def pure_shape_dice_loss(gw_mod: GlobalWorkspace2Domains, domain_latents: LatentsDomainGroupsT, raw_data: RawDomainGroupsT):
    visual_module = cast(VisualLatentDomainModule, gw_mod.domain_mods["v_latents"])
    visual_module.eval()

    with torch.no_grad():
        v_raw = raw_data[frozenset({'v_latents'})]['v_latents']
        x_original = visual_module.decode_images(v_raw)
        m_orig = get_mask_from_shapes_diff(x_original).detach()

    attr_input = domain_latents[frozenset({'attr'})]
    gw_latents_encoded = gw_mod.encode(attr_input)
    
    gw_decoded_latents = gw_mod.decode(gw_latents_encoded['attr'], domains={'v_latents'})
    
    x_recons = visual_module.decode_images(gw_decoded_latents['v_latents'])
    m_pred = get_mask_from_shapes_diff(x_recons)
    
    intersection = (m_pred * m_orig).sum(dim=(1,2,3))
    union = m_pred.sum(dim=(1,2,3)) + m_orig.sum(dim=(1,2,3))
    
    loss = 1 - (2. * intersection / (union + 1e-6)).mean()
    
    return loss
