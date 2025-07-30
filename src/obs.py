import math
from typing import Any

import numpy as np
from rlgym_compat import common_values
from rlgym_compat.v1_game_state import V1GameState, V1PhysicsObject, V1PlayerData


class CustomObs:
    POS_STD = 2300
    ANG_STD = math.pi

    obs_size = 9 + 8 + 25 + 34

    def __init__(self, cars):
        self.obs_size += 31 * (cars - 1)

    def reset(self, initial_state: V1GameState):
        pass

    def build_obs(
        self, player: V1PlayerData, state: V1GameState, previous_action: np.ndarray
    ) -> Any:
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [
            ball.position / CustomObs.POS_STD,
            ball.linear_velocity / CustomObs.POS_STD,
            ball.angular_velocity / CustomObs.ANG_STD,
            previous_action,
            pads,
        ]

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Extra info
            team_obs.extend(
                [
                    (other_car.position - player_car.position) / CustomObs.POS_STD,
                    (other_car.linear_velocity - player_car.linear_velocity)
                    / CustomObs.POS_STD,
                ]
            )

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _add_player_to_obs(
        self, obs: list, player: V1PlayerData, ball: V1PhysicsObject, inverted: bool
    ):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        obs.extend(
            [
                rel_pos / CustomObs.POS_STD,
                rel_vel / CustomObs.POS_STD,
                player_car.position / CustomObs.POS_STD,
                player_car.forward(),
                player_car.up(),
                player_car.linear_velocity / CustomObs.POS_STD,
                player_car.angular_velocity / CustomObs.ANG_STD,
                [
                    player.boost_amount,
                    int(player.on_ground),
                    int(player.has_flip),
                    int(player.is_demoed),
                ],
            ]
        )

        return player_car
