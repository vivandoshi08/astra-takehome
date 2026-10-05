"""Evaluate a neat-python genome as a plain-numpy network.

This is the rig, so you don't have to reimplement genome execution before you
can get to the actual question. It gives you:

    net = GenomeNetwork(genome, config)
    p      = net.forward(X)          # (N, 2) -> (N,) output-node activation
    theta  = net.get_params()        # flat float vector: weights then biases
    net.set_params(theta)            # load a vector back in
    net.write_back(genome)           # copy current params into the genome's genes
    L      = net.loss(X, y)          # mean binary cross-entropy of forward(X) vs y
    L, g   = net.loss_and_grad(X, y) # ... and dL/dtheta, in get_params() order

`forward` is written against `self.weights` / `self.biases`, so you can
differentiate through it however you like — autograd, torch, JAX, or by hand.
`loss_and_grad` does it by hand (reverse mode over `eval_order`);
test_gradients.py checks it against finite differences of forward().

Feed-forward genomes only (the config ships with `feed_forward = True`).

Changes from the shipped rig (see audit_scaffold.py for the evidence):
- `SigmoidOutputGenome` keeps the output node's activation at sigmoid. As
  shipped it mutated to tanh/relu in about half of all genomes, while
  `predict` thresholds at 0.5 and a likelihood loss needs a probability.
- `write_back` clamps to the config's weight/bias range, the same range
  `lower` / `upper` expose for refinement to project onto. As shipped, a value
  past the range was stored and then snapped back on its first mutation.
- `champion_net` rebuilds a genome's network with the parameters that earned
  its fitness (`genome.refined_params`), so a Baldwinian champion is not
  evaluated with the unrefined weights its genome still carries.

Activations here are the plain functions (sigmoid(z), tanh(z), relu(z), z,
sin(z), exp(-z^2)); neat-python's own `neat.nn` uses sigmoid(5z), tanh(2.5z),
sin(5z) and exp(-5z^2), so never evaluate with `neat.nn`. neat-python only
needs an activation's *name* (mutation picks among `activation_options`);
'sin' and 'gauss' are both in its built-in registry.
"""

from __future__ import annotations

import math

import neat
import numpy as np
from neat.graphs import feed_forward_layers


# gauss clips its input here before squaring (see _act).
GAUSS_CLIP = 40.0


def _act(name: str, x: np.ndarray) -> np.ndarray:
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "identity":
        return x
    if name == "sin":
        return np.sin(x)
    if name == "gauss":
        # Clipped only so x*x cannot overflow: exp(-x^2) is already exactly
        # 0.0 for |x| > 27.3, so the clip at 40 changes no value.
        c = np.clip(x, -GAUSS_CLIP, GAUSS_CLIP)
        return np.exp(-c * c)
    raise ValueError(f"unsupported activation {name!r} — see network.py")


def _act_grad(name: str, x: np.ndarray) -> np.ndarray:
    """d _act(name, x) / dx, elementwise.

    sigmoid is flat outside (-60, 60) because _act clips there; relu takes
    subgradient 0 at x <= 0. Written with exp(-|x|) so nothing overflows.
    gauss: -2x exp(-x^2), with the same clip as _act (the value there is
    already 0, so no kink is visible).
    """
    if name == "sigmoid":
        e = np.exp(-np.abs(x))
        return np.where(np.abs(x) < 60, e / (1.0 + e) ** 2, 0.0)
    if name == "tanh":
        e = np.exp(-2.0 * np.abs(x))
        return 4.0 * e / (1.0 + e) ** 2
    if name == "relu":
        return (x > 0).astype(float)
    if name == "identity":
        return np.ones_like(x)
    if name == "sin":
        return np.cos(x)
    if name == "gauss":
        c = np.clip(x, -GAUSS_CLIP, GAUSS_CLIP)
        return -2.0 * c * np.exp(-c * c)
    raise ValueError(f"unsupported activation {name!r} — see network.py")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Unclipped logistic sigmoid, without overflow for any z."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0, e) / (1.0 + e)


class GenomeNetwork:
    """A genome's topology, with its parameters exposed as a flat vector."""

    def __init__(self, genome, config):
        gc = config.genome_config
        self.input_keys = list(gc.input_keys)
        self.output_keys = list(gc.output_keys)

        self.conns = [k for k, c in genome.connections.items() if c.enabled]
        self.weights = np.array(
            [genome.connections[k].weight for k in self.conns], dtype=float
        )

        # neat-python 2.0 returns (layers, required); older versions returned
        # just the layer list. Its docstring still describes the old shape.
        result = feed_forward_layers(self.input_keys, self.output_keys,
                                     self.conns)
        layers = result[0] if isinstance(result, tuple) else result
        self.eval_order = [n for layer in layers for n in sorted(layer)]

        self.node_keys = list(self.eval_order)
        self.biases = np.array(
            [genome.nodes[n].bias for n in self.node_keys], dtype=float
        )
        self.activations = [genome.nodes[n].activation for n in self.node_keys]
        self._bias_ix = {n: i for i, n in enumerate(self.node_keys)}
        self._incoming = {
            n: [(i, src) for i, (src, dst) in enumerate(self.conns) if dst == n]
            for n in self.node_keys
        }

        # Per-parameter bounds, in get_params() order: the range NEAT's own
        # mutation operators clamp to.
        nw, nb = len(self.weights), len(self.biases)
        self.lower = np.r_[np.full(nw, gc.weight_min_value),
                           np.full(nb, gc.bias_min_value)]
        self.upper = np.r_[np.full(nw, gc.weight_max_value),
                           np.full(nb, gc.bias_max_value)]

    # ---- parameters -------------------------------------------------------
    @property
    def n_params(self) -> int:
        return len(self.weights) + len(self.biases)

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.weights, self.biases])

    def set_params(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_params,):
            raise ValueError(f"expected {self.n_params} params, got {theta.shape}")
        k = len(self.weights)
        self.weights, self.biases = theta[:k].copy(), theta[k:].copy()

    def write_back(self, genome) -> int:
        """Copy current parameters back into the genome's genes.

        Values are clamped to the config's weight/bias range. Returns how many
        had to be clamped: anything but 0 means the genome now differs from
        the network that was evaluated.
        """
        theta = self.get_params()
        clipped = np.clip(theta, self.lower, self.upper)
        k = len(self.weights)
        for c, w in zip(self.conns, clipped[:k]):
            genome.connections[c].weight = float(w)
        for n, b in zip(self.node_keys, clipped[k:]):
            genome.nodes[n].bias = float(b)
        return int((clipped != theta).sum())

    # ---- evaluation -------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Output-node activation for a batch. X is (N, 2); returns (N,).

        Not a logit: with a sigmoid output (SigmoidOutputGenome) it is
        P(class 1), and `predict` thresholds it at 0.5.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        values = {k: X[:, i] for i, k in enumerate(self.input_keys)}
        zeros = np.zeros(len(X))
        for node in self.eval_order:
            total = self.biases[self._bias_ix[node]] * np.ones(len(X))
            for ci, src in self._incoming[node]:
                total = total + self.weights[ci] * values.get(src, zeros)
            values[node] = _act(self.activations[self._bias_ix[node]], total)
        return values.get(self.output_keys[0], zeros)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.forward(X) > 0.5).astype(float)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())

    # ---- loss and gradient ------------------------------------------------
    # The loss is mean binary cross-entropy, computed from the output node's
    # pre-activation z as mean(softplus(z) - y*z): exact for every z, where
    # log(forward(X)) would round to log(0). forward() clips z to +-60 inside
    # its sigmoid; the loss does not, so it keeps a gradient on a point the
    # network is confidently wrong about. Below |z| = 60 the two agree exactly.

    def _output_key(self):
        out = self.output_keys[0]
        if out not in self._bias_ix:
            raise ValueError(f"output node {out} is not in eval_order, so "
                             "forward() returns 0 for every point; no loss")
        act = self.activations[self._bias_ix[out]]
        if act != "sigmoid":
            raise ValueError(f"the loss needs a sigmoid output node, got {act!r}: "
                             "build configs with load_config() (SigmoidOutputGenome)")
        return out

    def _pass(self, X: np.ndarray):
        """forward(), keeping each node's pre-activation.

        Same arithmetic as forward(). Returns (values, pre, live): `live[node]`
        lists the incoming (connection index, source) pairs whose source had a
        value when the node was evaluated; forward() reads any other source
        as 0, so its weight gets no gradient.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        values = {k: X[:, i] for i, k in enumerate(self.input_keys)}
        pre, live = {}, {}
        for node in self.eval_order:
            j = self._bias_ix[node]
            total = self.biases[j] * np.ones(len(X))
            live[node] = [(ci, s) for ci, s in self._incoming[node] if s in values]
            for ci, src in live[node]:
                total = total + self.weights[ci] * values[src]
            pre[node] = total
            values[node] = _act(self.activations[j], total)
        return values, pre, live

    @staticmethod
    def _bce(z: np.ndarray, y) -> tuple[float, np.ndarray]:
        y = np.asarray(y, dtype=float)
        if y.shape != z.shape:
            raise ValueError(f"y has shape {y.shape}, expected {z.shape}")
        return float(np.mean(np.logaddexp(0.0, z) - y * z)), y

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean binary cross-entropy of forward(X) against labels y in {0, 1}."""
        out = self._output_key()
        _, pre, _ = self._pass(X)
        return self._bce(pre[out], y)[0]

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """(loss(X, y), its gradient w.r.t. get_params()), by reverse mode.

        Parameters that cannot reach the output get exactly 0: connections
        into nodes outside eval_order, and weights whose source forward()
        reads as 0. Does not change the network's parameters.
        """
        out = self._output_key()
        values, pre, live = self._pass(X)
        z = pre[out]
        loss, y = self._bce(z, y)

        nw = len(self.weights)
        grad = np.zeros(self.n_params)
        adj = {}  # node -> dL/d(node's value), summed over the nodes it feeds
        for node in reversed(self.eval_order):
            j = self._bias_ix[node]
            if node == out:
                delta = (_sigmoid(z) - y) / len(z)  # dL/dz
            elif node in adj:
                delta = adj.pop(node) * _act_grad(self.activations[j], pre[node])
            else:
                continue  # feeds nothing that reaches the output
            grad[nw + j] = delta.sum()
            for ci, src in live[node]:
                grad[ci] = delta @ values[src]
                if src in self._bias_ix:  # a node, not an input
                    adj[src] = adj.get(src, 0.0) + delta * self.weights[ci]
        return loss, grad


def complexity(genome) -> tuple[int, int]:
    """(node count, enabled connection count) — useful for a size penalty.

    Counts every enabled gene, including ones that never reach the output
    (about a third of enabled connections in a typical population).
    """
    nodes = len(genome.nodes)
    conns = sum(1 for c in genome.connections.values() if c.enabled)
    return nodes, conns


def champion_net(genome, config) -> GenomeNetwork:
    """The network a genome's fitness was earned with.

    If evaluation stored refined parameters on the genome
    (`genome.refined_params`), load them; otherwise use the genome's own
    weights. Lamarckian genomes already carry their refined weights; a
    Baldwinian genome does not, so building it with GenomeNetwork alone would
    test the unrefined network.
    """
    net = GenomeNetwork(genome, config)
    theta = getattr(genome, "refined_params", None)
    if theta is not None:
        net.set_params(theta)
    return net


class SigmoidOutputGenome(neat.DefaultGenome):
    """DefaultGenome whose output nodes always use sigmoid.

    Hidden nodes still mutate among `activation_options`.
    """

    @staticmethod
    def _pin_outputs(genome, config):
        for k in config.output_keys:
            genome.nodes[k].activation = "sigmoid"

    def configure_new(self, config):
        super().configure_new(config)
        self._pin_outputs(self, config)

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self._pin_outputs(self, config)

    def mutate(self, config):
        super().mutate(config)
        self._pin_outputs(self, config)


def load_config(path) -> neat.Config:
    """neat.Config that builds SigmoidOutputGenome genomes.

    neat.Config reads the ini section named after the genome class, so parse
    with DefaultGenome (the [DefaultGenome] section, same parameters) and then
    swap the type: Population and reproduction create genomes through
    config.genome_type.
    """
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))
    config.genome_type = SigmoidOutputGenome
    return config
