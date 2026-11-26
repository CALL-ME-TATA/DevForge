"""Action masking from observation fields (explicitly allowed by docs/RULES.md section 1).

Used identically by the submitted agent and by the training workers, so the network is trained under
exactly the action rules it is evaluated with. The network still chooses every action; a mask can
only release a button, never press one.
"""
import numpy as np

ENEMY_TYPES = (1, 2, 3)  # enemy_melee, enemy_hitscan, enemy_projectile (doom_env.spaces.ENTITY_TYPES)
MOVE_BACKWARD = 1  # button index (docs/CONTRACT.md section 3)


def enemy_in_view(obs) -> bool:
    ve = obs["visible_entities"]
    m = np.asarray(ve["mask"]).astype(bool)
    return bool(np.isin(np.asarray(ve["entity_type"])[m], ENEMY_TYPES).any())


def mask_actions(a, obs):
    """MOVE_BACKWARD is only available while an enemy is in view (backing off in a fight).
    Otherwise the button is released, so the agent travels facing where it is going: its rays,
    entity list and screen only cover the 90 degrees in front of it."""
    if a[MOVE_BACKWARD] and not enemy_in_view(obs):
        a = np.array(a, copy=True)
        a[MOVE_BACKWARD] = 0
    return a
