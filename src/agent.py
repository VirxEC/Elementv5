from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ncnn import Mat, Net  # type: ignore


def load_actor() -> Net:
    current_folder = Path(__file__).parent

    actor = Net()
    actor.load_param(str(current_folder / "model.ncnn.param"))
    actor.load_model(str(current_folder / "model.ncnn.bin"))
    return actor


# This version of a dataclass is optimized for performance
# frozen is set to True because this class shouldn't be modified after creation
# slots is set to True to reduce memory usage
# everything is set to False (other than init) to reduce the amount of code generated
@dataclass(repr=False, eq=False, match_args=False, frozen=True, slots=True)
class Agent:
    state_space: int

    actor = load_actor()

    def act(self, state: np.ndarray):
        # this line is taken from `torch/agent.py`,
        # with the tensor operations replaced by numpy operations
        state = state.astype(np.float32).reshape(
            -1, self.state_space
        )  # 1st dimension is batch number

        with self.actor.create_extractor() as ex:
            # the below is taken from `ncnn/model_ncnn.py`
            ex.input("in0", Mat(state))
            _, probs_cat = ex.extract("out0")
            _, probs_ber = ex.extract("out1")

        # the below is taken from `torch/agent.py`,
        # with the tensor operations replaced by numpy operations
        actions_cat = np.argmax(np.array(probs_cat), axis=2)
        actions_ber = np.argmax(np.array(probs_ber), axis=2)

        actions = np.concatenate([actions_cat, actions_ber], 1)
        return actions
