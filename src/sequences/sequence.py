import numpy as np
from rlgym_compat.v1_game_state import V1GameState, V1PlayerData


class Sequence:
    def is_valid(self, player: V1PlayerData, game_state: V1GameState) -> bool:
        raise NotImplementedError()

    def get_action(
        self, player: V1PlayerData, game_state: V1GameState, prev_action: np.ndarray
    ) -> list:
        raise NotImplementedError()
