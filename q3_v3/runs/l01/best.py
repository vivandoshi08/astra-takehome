import math

def _render(cx, cy, a, b, th, amp):
    ct, st = math.cos(th), math.sin(th)
    out = []
    for i in range(16):
        for j in range(16):
            dx, dy = j - cx, i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = math.sqrt((u / a) ** 2 + (v / b) ** 2)
            grad = math.sqrt((u / (a * a)) ** 2 + (v / (b * b)) ** 2)
            edge = 0.5 + (1.0 - rho) / max(grad, 1e-10)
            out.append(amp * max(0.0, min(1.0, edge)))
    return out

def encode(img):
    bounds = ((0.0, 15.0), (0.0, 15.0), (1.3, 4.8),
              (1.3, 4.8), (-math.pi / 2, math.pi / 2))

    def norm(q):
        r = q[:]
        for k in range(4):
            r[k] = max(bounds[k][0], min(bounds[k][1], r[k]))
        r[4] = (r[4] + math.pi / 2) % math.pi - math.pi / 2
        return r

    def loss(q):
        base = _render(q[0], q[1], q[2], q[3], q[4], 1.0)
        den = sum(v * v for v in base)
        amp = max(0.4, min(1.1, sum(x * v for x, v in zip(img, base)) / max(den, 1e-12)))
        err = sum((x - amp * v) ** 2 for x, v in zip(img, base))
        return err, amp

    starts = []
    for cut in (0.08, 0.16, 0.24, 0.32):
        w = [max(0.0, x - cut) for x in img]
        mass = sum(w)
        if mass <= 1e-8:
            continue
        cx = sum((k % 16) * v for k, v in enumerate(w)) / mass
        cy = sum((k // 16) * v for k, v in enumerate(w)) / mass
        xx = yy = xy = 0.0
        for k, v in enumerate(w):
            dx, dy = k % 16 - cx, k // 16 - cy
            xx += v * dx * dx
            yy += v * dy * dy
            xy += v * dx * dy
        xx, yy, xy = xx / mass, yy / mass, xy / mass
        d = math.sqrt(max(0.0, (xx - yy) ** 2 + 4.0 * xy * xy))
        th = 0.5 * math.atan2(2.0 * xy, xx - yy)
        a = max(1.3, min(4.8, 2.0 * math.sqrt(max(0.2, (xx + yy + d) / 2.0))))
        b = max(1.3, min(4.8, 2.0 * math.sqrt(max(0.2, (xx + yy - d) / 2.0))))
        starts.extend(([cx, cy, a, b, th],
                       [cx, cy, b, a, th + math.pi / 2]))

    if not starts:
        starts = [[7.5, 7.5, 2.8, 2.0, 0.0]]

    best_q, best_err = starts[0], float("inf")
    for start in starts:
        q = norm(start)
        err, amp = loss(q)
        steps = [0.35, 0.35, 0.32, 0.32, 0.16]

        # Pattern search with both single-coordinate and coupled proposals.
        for _ in range(45):
            improved = True
            while improved:
                improved = False
                for k in range(5):
                    for sign in (-1.0, 1.0):
                        trial = q[:]
                        trial[k] += sign * steps[k]
                        trial = norm(trial)
                        e, ap = loss(trial)
                        if e < err:
                            q, err, amp = trial, e, ap
                            improved = True

            for p, r in ((0, 1), (2, 3), (0, 2), (1, 3)):
                for s in (-1.0, 1.0):
                    trial = q[:]
                    trial[p] += s * steps[p]
                    trial[r] += s * steps[r]
                    trial = norm(trial)
                    e, ap = loss(trial)
                    if e < err:
                        q, err, amp = trial, e, ap

            steps = [s * 0.68 for s in steps]
            if max(steps) < 0.001:
                break

        if err < best_err:
            best_q, best_err = q[:], err

    _, amp = loss(best_q)
    return best_q + [amp]

def decode(z):
    return _render(z[0], z[1], z[2], z[3], z[4], z[5])
