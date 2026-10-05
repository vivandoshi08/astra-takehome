"""Evaluate a neat-python genome as a plain-numpy network.

This is the rig, so you don't have to reimplement genome execution before you
can get to the actual question. It gives you:

    net = GenomeNetwork(genome, config)
    p      = net.forward(X)          # (N, 2) -> (N,) output-node activation
    theta  = net.get_params()        # flat float vector: weights then biases
    net.set_params(theta)            # load a vector back in
    net.write_back(genome)           # copy current params into the genome's genes

`forward` is written against `self.weights` / `self.biases`, so you can
differentiate through it however you like — autograd, torch, JAX, or by hand.

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

Activations here are the plain functions (sigmoid(z), tanh(z)); neat-python's
own `neat.nn` uses sigmoid(5z) and tanh(2.5z), so never evaluate with `neat.nn`.
"""

from __future__ import annotations

import math

import neat
import numpy as np
from neat.graphs import feed_forward_layers


def _act(name: str, x: np.ndarray) -> np.ndarray:
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "identity":
        return x
    raise ValueError(f"unsupported activation {name!r} — see network.py")


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
