custom_weights_4mod = {
    'contrastive_cat_and_color': 0.1,
  'contrastive_cat_and_v_latents': 0.1,
  'contrastive_color_and_v_latents': 0.1,
  'contrastive_position_and_cat': 0.1,
  'contrastive_position_and_color': 0.1,
  'contrastive_position_and_v_latents': 0.1,
  
# --- Source: CAT ---
    'cycle_cat_through_color': 1.0,
    'cycle_cat_through_position': 0.0,
    'cycle_cat_through_v_latents': 0.0,
    'cycle_cat_through_color/position': 0.0,
    'cycle_cat_through_color/v_latents': 0.0,
    'cycle_cat_through_position/v_latents': 0.0,
    'cycle_cat_through_color/position/v_latents': 0.0,

    # --- Source: COLOR ---
    'cycle_color_through_cat': 1.0,
    'cycle_color_through_position': 0.0,
    'cycle_color_through_v_latents': 0.0,
    'cycle_color_through_cat/position': 0.0,
    'cycle_color_through_cat/v_latents': 0.0,
    'cycle_color_through_position/v_latents': 0.0,
    'cycle_color_through_cat/position/v_latents': 0.0,

    # --- Source: POSITION ---
    'cycle_position_through_cat': 0.0,
    'cycle_position_through_color': 0.0,
    'cycle_position_through_v_latents': 0.0,
    'cycle_position_through_cat/color': 0.0,
    'cycle_position_through_cat/v_latents': 0.0,
    'cycle_position_through_color/v_latents': 0.0,
    'cycle_position_through_cat/color/v_latents': 0.0,

    # --- Source: V_LATENTS ---
    'cycle_v_latents_through_cat': 0.0,
    'cycle_v_latents_through_color': 0.0,
    'cycle_v_latents_through_position': 0.0,
    'cycle_v_latents_through_cat/color': 0.0,
    'cycle_v_latents_through_cat/position': 0.0,
    'cycle_v_latents_through_color/position': 0.0,
    'cycle_v_latents_through_cat/color/position': 0.0,
    
  'demi_cycle_cat': 1.0,
  'demi_cycle_color': 1.0,
  'demi_cycle_position': 1.0,
  'demi_cycle_v_latents': 1.0,

  'translation_color_to_cat': 0.0,
  'translation_position_to_cat': 0.0,
  'translation_v_latents_to_cat': 1.0,
  'translation_color/position_to_cat': 0.0,
  'translation_color/v_latents_to_cat': 0.0,
  'translation_position/v_latents_to_cat': 0.0,
  'translation_color/position/v_latents_to_cat': 0.0,

  'translation_cat_to_color': 0.0,
  'translation_position_to_color': 0.0,
  'translation_v_latents_to_color': 1.0,
  'translation_cat/position_to_color': 0.0,
  'translation_cat/v_latents_to_color': 0.0,
  'translation_position/v_latents_to_color': 0.0,
  'translation_cat/position/v_latents_to_color': 0.0,

  'translation_cat_to_position': 0.0,
  'translation_color_to_position': 0.0,
  'translation_v_latents_to_position': 1.0,
  'translation_cat/color_to_position': 0.0,
  'translation_cat/v_latents_to_position': 0.0,
  'translation_color/v_latents_to_position': 0.0,
  'translation_cat/color/v_latents_to_position': 0.0,

  'translation_cat_to_v_latents': 0,
  'translation_color_to_v_latents': 0,
  'translation_position_to_v_latents': 0,
  'translation_cat/color_to_v_latents': 1.0,
  'translation_cat/position_to_v_latents': 1.0,
  'translation_color/position_to_v_latents': 1.0,
  'translation_cat/color/position_to_v_latents': 1.0
  }

custom_weights_3mod = {
    # --- DEMI-CYCLE (Reconstruction directe) ---
    'demi_cycle_attr': 1.0,
    'demi_cycle_color': 1.0,
    'demi_cycle_v_latents': 1.0,

    # --- TRANSLATIONS (N-to-1) ---
    # Cible: ATTR
    'translation_color_to_attr': 1.0,
    'translation_v_latents_to_attr': 1.0,
    'translation_color/v_latents_to_attr': 1.0,

    # Cible: COLOR
    'translation_attr_to_color': 1.0,
    'translation_v_latents_to_color': 1.0,
    'translation_attr/v_latents_to_color': 1.0,

    # Cible: V_LATENTS
    'translation_attr_to_v_latents': 1.0,
    'translation_color_to_v_latents': 1.0,
    'translation_attr/color_to_v_latents': 1.0,

    # --- CYCLES (1-through-N) ---
    # Source: ATTR
    'cycle_attr_through_color': 1.0,
    'cycle_attr_through_v_latents': 1.0,
    'cycle_attr_through_color/v_latents': 1.0,

    # Source: COLOR
    'cycle_color_through_attr': 1.0,
    'cycle_color_through_v_latents': 1.0,
    'cycle_color_through_attr/v_latents': 1.0,

    # Source: V_LATENTS
    'cycle_v_latents_through_attr': 1.0,
    'cycle_v_latents_through_color': 1.0,
    'cycle_v_latents_through_attr/color': 1.0,

    # --- CONTRASTIVE (Pairs) ---
    'contrastive_attr_and_color': 1.0,
    'contrastive_attr_and_v_latents': 1.0,
    'contrastive_color_and_v_latents': 1.0,
}