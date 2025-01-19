import numpy as np
from rlgym_compat.v1_game_state import V1GameState, V1PlayerData

from .sequence import Sequence


class Speedflip(Sequence):
    state = "align"
    initial_angle = np.pi / 16
    valid_distance = 600

    def __init__(self, player: V1PlayerData):
        car = player.car_data if player.team_num == 0 else player.inverted_car_data
        self.initial_forward = car.forward().copy()
        self.initial_position = car.position.copy()
        self.direction = 1 if car.position[0] > 0 else -1

        if abs(car.position[0]) < 25:
            self.yaw_strength = 1
            self.drive_distance = 290
            self.initial_angle = np.pi / 22
        elif abs(car.position[0]) < 500:
            self.yaw_strength = 0.6
            self.drive_distance = 330
            self.direction = -self.direction
        else:
            self.yaw_strength = 0.4
            self.drive_distance = 240

    def is_valid(self, player: V1PlayerData, game_state: V1GameState) -> bool:
        ball = game_state.ball if player.team_num == 0 else game_state.inverted_ball
        car = player.car_data if player.team_num == 0 else player.inverted_car_data
        ball_dist = np.linalg.norm(ball.position - car.position)
        return self.state != "done" and ball_dist > self.valid_distance  # type: ignore

    def get_action(
        self, player: V1PlayerData, game_state: V1GameState, prev_action: np.ndarray
    ) -> list:
        car = player.car_data if player.team_num == 0 else player.inverted_car_data

        if self.state == "align":
            angle = np.arccos(np.dot(car.forward(), self.initial_forward))
            if angle < self.initial_angle:
                return [1, -self.direction, 0, 0, 0, 0, 1, 0]
            else:
                self.state = "drive"

        if self.state == "drive":
            distance = np.linalg.norm(car.position - self.initial_position)
            if distance < self.drive_distance:
                return [1, 0, 0, 0, 0, 0, 1, 0]
            else:
                self.state = "first_jump"

        if self.state == "first_jump":
            self.state = "start_flip"
            return [1, 0, 0, 0, 0, 1, 1, 0]

        if self.state == "start_flip":
            if player.on_ground:
                # release jump
                return [1, 0, 0, 0, self.direction, 0, 1, 0]
            else:
                # dodge
                self.state = "cancel_flip"
                return [1, 0, -1, 0, self.direction, 1, 1, 0]

        if self.state == "cancel_flip":
            boost = 1 if np.linalg.norm(car.linear_velocity) < 2295 else 0

            if not player.on_ground:
                return [
                    1,
                    0,
                    1,
                    self.yaw_strength * self.direction,
                    self.direction,
                    0,
                    boost,
                    0,
                ]
            else:
                # landed
                self.state = "done"
                return [1, 0, 0, 0, 0, 0, 1, 0]

        raise AssertionError("State machine didn't return a value")
