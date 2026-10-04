"""Evaluate a neat-python genome as a plain-numpy network.

This is the rig, so you don't have to reimplement genome execution before you
can get to the actual question. It gives you:

    net = GenomeNetwork(genome, config)
    logits = net.forward(X)          # (N, 2) -> (N,)
    theta  = net.get_params()        # flat float vector: weights then biases
    net.set_params(theta)            # load a vector back in
    net.write_back(genome)           # copy current params into the genome's genes

`forward` is written against `self.weights` / `self.biases`, so you can
differentiate through it however you like — autograd, torch, JAX, or by hand.

Feed-forward genomes only (the config ships with `feed_forward = True`).
"""

from __future__ import annotations

import math

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

    def write_back(self, genome) -> None:
        """Copy current parameters back into the genome's genes."""
        for k, w in zip(self.conns, self.weights):
            genome.connections[k].weight = float(w)
        for n, b in zip(self.node_keys, self.biases):
            genome.nodes[n].bias = float(b)

    # ---- evaluation -------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Logits for a batch of inputs. X is (N, 2); returns (N,)."""
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
    """(node count, enabled connection count) — useful for a size penalty."""
    nodes = len(genome.nodes)
    conns = sum(1 for c in genome.connections.values() if c.enabled)
    return nodes, conns
