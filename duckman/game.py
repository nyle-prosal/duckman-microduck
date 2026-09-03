"""Game rules, scoring, phases and the event log. Owns the Sim and calls every policy each step."""
from dataclasses import dataclass
from .constants import CLOCK_S, LIVES, POWER_S, SCORE, CTRL_DT
from .world import Sim


@dataclass
class GameView:
    maze: object
    t: float
    clock_left: float
    lives: int
    score: int
    power_left: float
    phase: str            # play | down | reset_done
    duck_pos: dict
    duck_yaw: dict
    duck_cell: dict
    upright: dict
    coins_alive: dict     # cell -> bool
    pellets_alive: dict   # cell -> bool
    ghost_mode: dict      # prefix -> chase | frightened | eaten | home
    ghost_home: dict      # prefix -> cell
    events_this_step: list
    tagged: bool


class Game:
    IMMUNE_S = 1.5
    HOME_WAIT_S = 2.0

    def __init__(self, maze, seed, pac_policy, ghost_policies, log_positions=False):
        self.maze, self.seed = maze, seed
        self.sim = Sim(maze, seed)
        self.policies = {"P_": pac_policy, **{f"G{k}_": ghost_policies[k] for k in range(4)}}
        self.log_positions = log_positions
        self.view = None
        self.done = False

    def reset(self):
        self.sim.reset(self.seed)
        for p in self.policies.values():
            p.reset(self.seed)
        self.lives, self.score, self.power_left, self.phase, self.immune = LIVES, 0, 0.0, "play", 0.0
        self.coins = {n: True for n in self.sim.info["coins"]}
        self.pellets = {n: True for n in self.sim.info["pellets"]}
        self.ghost_mode = {f"G{k}_": "chase" for k in range(4)}
        self.home_t = {g: 0.0 for g in self.ghost_mode}
        self.ghost_home = {f"G{k}_": self.maze.ghost_starts[k] for k in range(4)}
        self.events, self.positions_log = [], []
        self.done, self.end_reason, self._tagged = False, None, False
        self.counts = dict(coins=0, pellets=0, ghosts=0, lives_lost=0)
        self.view = self._view([])
        return self.view

    @staticmethod
    def _cellname(n):
        a, b = n.split("_")[1:]
        return (int(a), int(b))

    def _view(self, events):
        d = self.sim.ducks
        return GameView(self.maze, self.sim.t, CLOCK_S - self.sim.t, self.lives, self.score, self.power_left, self.phase,
                        {p: x.pos()[:2] for p, x in d.items()}, {p: x.yaw() for p, x in d.items()},
                        {p: self.maze.cell(*x.pos()[:2]) for p, x in d.items()}, {p: x.upright() for p, x in d.items()},
                        {self._cellname(n): v for n, v in self.coins.items()},
                        {self._cellname(n): v for n, v in self.pellets.items()},
                        dict(self.ghost_mode), dict(self.ghost_home), events, self._tagged)

    def _event(self, name, ev, **data):
        ev.append(name)
        self.events.append((round(self.sim.t, 2), name, data))

    def _end(self, reason, ev):
        self.done = True
        self.end_reason = reason
        self._event("end", ev, reason=reason)

    def step(self):
        if self.done:
            return self.view
        for prefix, pol in self.policies.items():
            obs = {"proprio": self.sim.ducks[prefix].proprio(pol.last), "view": self.view, "prefix": prefix}
            self.sim.ducks[prefix].set_targets(pol.act(obs))
        self.sim.step_physics()
        ev = []
        self._tagged = False
        for n, alive in self.coins.items():
            if alive and not self.sim.token_upright(n):
                self.coins[n] = False
                self.score += SCORE["coin"]
                self.counts["coins"] += 1
                self._event("coin", ev, cell=self._cellname(n))
        for n, alive in self.pellets.items():
            if alive and not self.sim.token_upright(n):
                self.pellets[n] = False
                self.score += SCORE["pellet"]
                self.counts["pellets"] += 1
                self.power_left = POWER_S
                self._event("pellet", ev, cell=self._cellname(n))
                for g, m in self.ghost_mode.items():
                    if m == "chase":
                        self.ghost_mode[g] = "frightened"
        if self.power_left > 0:
            self.power_left = max(0.0, self.power_left - CTRL_DT)
            if self.power_left == 0:
                for g, m in self.ghost_mode.items():
                    if m == "frightened":
                        self.ghost_mode[g] = "chase"
        self.immune = max(0.0, self.immune - CTRL_DT)
        if self.phase == "play":
            for g in list(self.ghost_mode):
                if not self.sim.contact("P_", g):
                    continue
                if self.ghost_mode[g] == "frightened":
                    self.ghost_mode[g] = "eaten"
                    self.score += SCORE["ghost"]
                    self.counts["ghosts"] += 1
                    self._event("ghost_caught", ev, ghost=g)
                elif self.ghost_mode[g] == "chase" and self.immune == 0:
                    self.lives -= 1
                    self.counts["lives_lost"] += 1
                    self._tagged = True
                    self.phase = "down"
                    self._event("tag", ev, ghost=g, lives=self.lives)
                    for h in self.ghost_mode:
                        self.ghost_mode[h] = "home"
                    if self.lives == 0:
                        self._end("game_over", ev)
                    break
        for g, m in self.ghost_mode.items():
            at_home = self.maze.cell(*self.sim.ducks[g].pos()[:2]) == self.ghost_home[g]
            self.home_t[g] = self.home_t[g] + CTRL_DT if (m in ("eaten", "home") and at_home) else 0.0
            if m == "eaten" and self.home_t[g] >= self.HOME_WAIT_S:
                self.ghost_mode[g] = "chase"
        if self.phase == "down" and all(self.ghost_mode[g] == "home" and self.home_t[g] >= self.HOME_WAIT_S
                                        for g in self.ghost_mode):
            self.phase = "reset_done"
        if self.phase == "reset_done" and self.policies["P_"].ready():
            self.phase = "play"
            self.immune = self.IMMUNE_S
            self._event("resume", ev)
            for g in self.ghost_mode:
                self.ghost_mode[g] = "chase"
        if not self.done and not any(self.coins.values()):
            self.score += SCORE["clear"]
            self._end("cleared", ev)
        if not self.done and self.sim.t >= CLOCK_S - 1e-9:
            self._end("timeout", ev)
        if self.log_positions:
            self.positions_log.append(self.sim.root_positions())
        self.view = self._view(ev)
        return self.view

    def run(self, max_t=CLOCK_S):
        while not self.done and self.sim.t < max_t - 1e-9:
            self.step()
        return self.result()

    def result(self):
        return dict(seed=self.seed, score=self.score, t=round(self.sim.t, 2), end=self.end_reason, lives=self.lives,
                    cleared=self.end_reason == "cleared", **self.counts, events=self.events,
                    calls={p: {"policy": pol.calls, "gait": pol.gait_calls()} for p, pol in self.policies.items()},
                    strategy_calls=self.policies["P_"].strategy_calls,
                    action_bounds={p: pol.action_bounds() for p, pol in self.policies.items()},
                    labels={p: pol.label for p, pol in self.policies.items()})
