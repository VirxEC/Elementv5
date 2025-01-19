import numpy as np
from rlbot.flat import ControllerState, GamePacket, MatchPhase
from rlbot.managers import Bot
from rlgym_compat.v1_game_state import V1GameState

from agent import Agent
from obs import CustomObs
from sequences.speedflip import Speedflip


class Element(Bot):
    obs_builder = CustomObs(cars=2)
    agent = Agent(obs_builder.obs_size)
    action_trans = np.array([-1, -1, -1, -1, -1, 0, 0, 0])
    tick_skip = 8
    controls = ControllerState()
    action = np.zeros(8)
    update_action = True
    ticks = tick_skip
    prev_frame = 0
    kickoff_seq = None

    def initialize(self):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        self.game_state = V1GameState(self.field_info, self.match_config)
        self.logger.info("Element Ready - Index: %s", self.index)

    def get_output(self, packet: GamePacket) -> ControllerState:
        if len(packet.balls) == 0 or packet.match_info.match_phase in {
            MatchPhase.Inactive,
            MatchPhase.Paused,
            MatchPhase.Countdown,
            MatchPhase.Replay,
        }:
            return self.controls

        cur_frame = packet.match_info.frame_num
        delta = cur_frame - self.prev_frame
        self.prev_frame = cur_frame

        self.ticks += delta
        self.game_state.update(packet)

        if packet.match_info.match_phase == MatchPhase.Kickoff:
            try:
                player = self.game_state.players[self.index]
                teammates = [
                    p for p in self.game_state.players if p.team_num == self.team
                ]
                closest = min(
                    teammates,
                    key=lambda p: np.linalg.norm(
                        self.game_state.ball.position - p.car_data.position
                    ),
                )

                if self.kickoff_seq is None:
                    self.kickoff_seq = Speedflip(player)

                if player == closest and self.kickoff_seq.is_valid(
                    player, self.game_state
                ):
                    self.action = np.asarray(
                        self.kickoff_seq.get_action(
                            player, self.game_state, self.action
                        )
                    )
                    self.update_controls(self.action)
                    return self.controls
            except:
                self.logger.error(
                    "Element - Kickoff sequence failed, falling back to model"
                )
        else:
            self.kickoff_seq = None

        # We calculate the next action as soon as the prev action is sent
        # This gives you tick_skip ticks to do your forward pass
        if self.update_action:
            self.update_action = False

            # This model is 1v1, remove the extra players from the state
            player = self.game_state.players[self.index]

            teammates = [p for p in self.game_state.players if p.team_num == self.team]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            # Sort by distance to ball
            teammates.sort(
                key=lambda p: np.linalg.norm(
                    self.game_state.ball.position - p.car_data.position
                )
            )
            opponents.sort(
                key=lambda p: np.linalg.norm(
                    self.game_state.ball.position - p.car_data.position
                )
            )

            # Grab opponent in same "position" as Element relative to it's teammates
            opponent = opponents[min(teammates.index(player), len(opponents) - 1)]

            self.game_state.players = [player, opponent]

            obs = self.obs_builder.build_obs(player, self.game_state, self.action)
            self.action = self.agent.act(obs)[0] + self.action_trans
            # print(self.action)

        if self.ticks >= self.tick_skip - 1:
            self.update_controls(self.action)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_action = True

        return self.controls

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = 0 if action[5] > 0 else action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0


if __name__ == "__main__":
    Element("Rangler/Element").run(
        wants_match_communications=False, wants_ball_predictions=False
    )
