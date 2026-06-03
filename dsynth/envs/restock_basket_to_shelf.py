import torch
import numpy as np
from transforms3d.euler import euler2quat
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from dsynth.envs.pick_to_basket import PickToBasketContEnv


@register_env('RestockBasketToShelfContEnv', max_episode_steps=200000)
class RestockBasketToShelfContEnv(PickToBasketContEnv):
    """Pick target product from robot basket, place it back on shelf empty slot.

    This is the reverse of PickToBasket: the robot already has the target item
    in its basket (placed there from the shelf). The task is to pick it up from
    the basket and put it back on the shelf at its original position.
    """

    TARGET_PRODUCT_NAME = 'Duff Beer Can'
    ROBOT_INIT_POSE_RANDOM_ENABLED = False
    NUM_BASKET_DISTRACTORS = 0  # extra non-target items also placed in the basket

    # world-frame xy offsets (m) so basket items don't interpenetrate
    _BASKET_SLOTS = [(0.0, 0.0), (0.08, 0.08), (-0.08, -0.08),
                     (0.08, -0.08), (-0.08, 0.08)]

    def __init__(self, *args, num_basket_distractors=None, **kwargs):
        # set before super().__init__: it triggers reset() -> _initialize_episode
        if num_basket_distractors is not None:
            self.NUM_BASKET_DISTRACTORS = int(num_basket_distractors)
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        # actor names of every item placed in the basket, per scene. The solver
        # uses this to whitelist basket items in the collision matrix so the
        # gripper may reach the target without IK failing on neighbour items.
        self.basket_item_names = {}

        for scene_idx in range(self.num_envs):
            rng = self._batched_episode_rng[scene_idx]

            scene_target_products = self.target_products_df[
                self.target_products_df['scene_idx'] == scene_idx]
            target_actor_name = scene_target_products['actor_name'].iloc[0]

            # target goes in first, then a few distinct-category distractors
            actor_names = [target_actor_name]
            scene_products = self.products_df[self.products_df['scene_idx'] == scene_idx]
            target_pname = self.target_product_names[scene_idx]
            other_pnames = sorted(
                scene_products[scene_products['product_name'] != target_pname]['product_name'].unique())
            n_extra = min(self.NUM_BASKET_DISTRACTORS, len(other_pnames))
            for pname in (rng.choice(other_pnames, size=n_extra, replace=False) if n_extra else []):
                cand = scene_products[scene_products['product_name'] == pname]
                actor_names.append(cand['actor_name'].iloc[0])

            self.basket_item_names[scene_idx] = actor_names

            basket_center = self.calc_target_pose().p[scene_idx].clone()
            basket_center[0] += 0.05
            basket_center[2] += 0.08

            for k, actor_name in enumerate(actor_names):
                dx, dy = self._BASKET_SLOTS[k % len(self._BASKET_SLOTS)]
                pos = basket_center.clone()
                pos[0] += dx
                pos[1] += dy
                random_q = np.array(euler2quat(
                    (rng.random() - 0.5) * np.pi * 0.3,
                    (rng.random() - 0.5) * np.pi * 0.3,
                    rng.random() * 2 * np.pi))
                random_q = torch.from_numpy(random_q).float().unsqueeze(0).to(self.device)
                self.actors['products'][actor_name].set_pose(
                    Pose.create_from_pq(p=pos.unsqueeze(0), q=random_q))

    def evaluate(self):
        is_obj_placed = []
        for scene_idx in range(self.num_envs):
            scene_is_obj_placed = torch.tensor([False], device=self.device)
            scene_target_products_df = self.target_products_df[
                self.target_products_df['scene_idx'] == scene_idx]
            for actor_name in scene_target_products_df['actor_name']:
                actor = self.actors['products'][actor_name]
                if actor_name in self.products_initial_poses:
                    init_p = self.products_initial_poses[actor_name][:, :3]
                    cur_p = actor.pose.p
                    if torch.all(torch.isclose(cur_p, init_p, rtol=0.1, atol=0.1)):
                        scene_is_obj_placed = torch.tensor([True], device=self.device)
                        break
            is_obj_placed.append(scene_is_obj_placed)
        is_obj_placed = torch.cat(is_obj_placed)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "success": is_obj_placed & is_robot_static,
        }

    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            self.language_instructions.append(
                f'pick {self.target_product_names[scene_idx]} from basket and place on shelf')


@register_env('RestockBasketToShelfContNiveaEnv', max_episode_steps=200000)
class RestockBasketToShelfContNiveaEnv(RestockBasketToShelfContEnv):
    TARGET_PRODUCT_NAME = 'Nivea Body Milk'


@register_env('RestockBasketToShelfContFantaEnv', max_episode_steps=200000)
class RestockBasketToShelfContFantaEnv(RestockBasketToShelfContEnv):
    TARGET_PRODUCT_NAME = 'Fanta Sabor Naranja 2L'


@register_env('RestockBasketToShelfContStarsEnv', max_episode_steps=200000)
class RestockBasketToShelfContStarsEnv(RestockBasketToShelfContEnv):
    TARGET_PRODUCT_NAME = 'Nestle Honey Stars'


@register_env('RestockBasketToShelfContDuffEnv', max_episode_steps=200000)
class RestockBasketToShelfContDuffEnv(RestockBasketToShelfContEnv):
    TARGET_PRODUCT_NAME = 'Duff Beer Can'
