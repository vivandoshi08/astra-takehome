"""A variational autoencoder in plain numpy, backprop written out by hand.

This is the learned incumbent the loop is compared against (see
baselines.py). It is provided, it is correct, and reimplementing it is not
the question — read it once so you know what it does, then use it as an
instrument: it is cheap enough to retrain with any width, latent size,
beta, or training set you want to probe.

Architecture: MLP encoder [ambient, *hidden, 2*K] producing (mu, logvar),
reparameterized sample z = mu + exp(logvar/2) * eps, MLP decoder
[K, *reversed(hidden), ambient]. tanh on hidden layers, linear elsewhere.

Loss per batch: mean squared reconstruction error against the target
images + beta * KL(q(z|x) || N(0, I)) / ambient  (the KL is divided by the
pixel count so beta is comparable across image sizes). Optimiser is Adam.
Deterministic given `seed`. At evaluation time encode() returns mu and
reconstruct() is deterministic.

    model = VAE(ambient=256, hidden=(256,), latent=3, beta=0.05, seed=0)
    model.train(X, targets=X, epochs=2000)
    Z = model.encode(X)          # (N, ambient) -> (N, K)   [returns mu]
    R = model.reconstruct(X)     # (N, ambient) -> (N, ambient)
    model.save(path); VAE.load(path)
"""

from __future__ import annotations

import numpy as np


class VAE:
    def __init__(self, ambient: int, hidden=(256,), latent: int = 3,
                 beta: float = 0.05, seed: int = 0):
        hidden = tuple(hidden)
        self.ambient, self.hidden, self.latent = ambient, hidden, latent
        self.beta = float(beta)
        self.seed = seed
        enc_sizes = [ambient, *hidden, 2 * latent]
        dec_sizes = [latent, *reversed(hidden), ambient]
        rng = np.random.default_rng(seed)

        def init(sizes):
            W = [rng.normal(0.0, np.sqrt(2.0 / (m + n)), size=(m, n))
                 for m, n in zip(sizes[:-1], sizes[1:])]
            b = [np.zeros(n) for n in sizes[1:]]
            return W, b

        self.eW, self.eb = init(enc_sizes)
        self.dW, self.db = init(dec_sizes)
        self._params = [*self.eW, *self.eb, *self.dW, *self.db]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._t = 0
        self._rng = np.random.default_rng(seed + 1_000_003)

    # ------------------------------------------------------------- forward
    def _enc_forward(self, X):
        acts = [np.atleast_2d(X)]
        h = acts[0]
        last = len(self.eW) - 1
        for i, (w, v) in enumerate(zip(self.eW, self.eb)):
            z = h @ w + v
            h = z if i == last else np.tanh(z)
            acts.append(h)
        mu, logvar = h[:, :self.latent], np.clip(h[:, self.latent:], -10, 10)
        return acts, mu, logvar

    def _dec_forward(self, Z):
        acts = [np.atleast_2d(Z)]
        h = acts[0]
        last = len(self.dW) - 1
        for i, (w, v) in enumerate(zip(self.dW, self.db)):
            z = h @ w + v
            h = z if i == last else np.tanh(z)
            acts.append(h)
        return acts

    def encode(self, X):
        _, mu, _ = self._enc_forward(X)
        return mu

    def decode(self, Z):
        return self._dec_forward(Z)[-1]

    def reconstruct(self, X):
        return self.decode(self.encode(X))

    # ------------------------------------------------------------ backward
    def _mlp_backward(self, W, acts, d_out):
        """Backprop d_out through an MLP given its activations.
        Returns (gW, gb, d_input)."""
        last = len(W) - 1
        gW = [None] * len(W)
        gb = [None] * len(W)
        d = d_out
        for i in range(last, -1, -1):
            a_next = acts[i + 1]
            dz = d if i == last else d * (1.0 - a_next ** 2)
            gW[i] = acts[i].T @ dz
            gb[i] = dz.sum(axis=0)
            d = dz @ W[i].T
        return gW, gb, d

    def _step(self, X, target, lr):
        n = len(X)
        e_acts, mu, logvar = self._enc_forward(X)
        eps = self._rng.standard_normal(mu.shape)
        std = np.exp(0.5 * logvar)
        z = mu + std * eps
        d_acts = self._dec_forward(z)
        out = d_acts[-1]

        recon = float(np.mean((out - target) ** 2))
        kl = 0.5 * np.sum(mu ** 2 + np.exp(logvar) - 1.0 - logvar) / (
            n * self.ambient)
        loss = recon + self.beta * kl

        # decoder gradients
        d_out = 2.0 * (out - target) / out.size
        gdW, gdb, dz_dec = self._mlp_backward(self.dW, d_acts, d_out)

        # gradients into mu / logvar: reconstruction path + KL path
        c = self.beta * 0.5 / (n * self.ambient)
        d_mu = dz_dec + c * 2.0 * mu
        d_logvar = dz_dec * eps * std * 0.5 + c * (np.exp(logvar) - 1.0)
        d_enc_out = np.concatenate([d_mu, d_logvar], axis=1)
        geW, geb, _ = self._mlp_backward(self.eW, e_acts, d_enc_out)

        grads = [*geW, *geb, *gdW, *gdb]
        self._t += 1
        b1, b2, ee = 0.9, 0.999, 1e-8
        c1 = 1.0 - b1 ** self._t
        c2 = 1.0 - b2 ** self._t
        for p, g, m, v in zip(self._params, grads, self._m, self._v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g ** 2
            p -= lr * (m / c1) / (np.sqrt(v / c2) + ee)
        return loss, recon, kl

    # ------------------------------------------------------------ training
    def train(self, X, targets=None, epochs: int = 2000, batch: int = 128,
              lr: float = 1e-3, callback=None):
        """Train to reconstruct `targets` (default: X itself) from X.
        Returns per-epoch mean loss history. Deterministic given seed."""
        X = np.asarray(X, dtype=float)
        T = X if targets is None else np.asarray(targets, dtype=float)
        n = len(X)
        history = []
        for epoch in range(epochs):
            order = self._rng.permutation(n)
            losses = []
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                losses.append(self._step(X[idx], T[idx], lr)[0])
            history.append(float(np.mean(losses)))
            if callback is not None:
                callback(epoch, history[-1], self)
        return history

    # ---------------------------------------------------------------- io
    def save(self, path):
        arrays = {}
        for tag, Ws, bs in (("e", self.eW, self.eb), ("d", self.dW, self.db)):
            for i, (w, v) in enumerate(zip(Ws, bs)):
                arrays[f"{tag}W{i}"] = w
                arrays[f"{tag}b{i}"] = v
        np.savez(path,
                 meta=np.array([self.ambient, self.latent, self.seed]),
                 beta=np.array([self.beta]),
                 hidden=np.array(self.hidden), **arrays)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        ambient, latent, seed = (int(x) for x in d["meta"])
        model = cls(ambient=ambient,
                    hidden=tuple(int(h) for h in d["hidden"]),
                    latent=latent, beta=float(d["beta"][0]), seed=seed)
        for i in range(len(model.eW)):
            model.eW[i] = d[f"eW{i}"]
            model.eb[i] = d[f"eb{i}"]
        for i in range(len(model.dW)):
            model.dW[i] = d[f"dW{i}"]
            model.db[i] = d[f"db{i}"]
        model._params = [*model.eW, *model.eb, *model.dW, *model.db]
        return model
