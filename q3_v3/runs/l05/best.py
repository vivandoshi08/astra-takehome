import math

def _shape(cx, cy, a, b, th, amp):
    ct, st = math.cos(th), math.sin(th)
    out = []
    for i in range(16):
        for j in range(16):
            dx, dy = j - cx, i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = math.sqrt((u / a) ** 2 + (v / b) ** 2)
            p = 0.5 + (1.0 - rho) * min(a, b)
            out.append(amp * max(0.0, min(1.0, p)))
    return out

def encode(img):
    # Bright-pixel moments give a useful starting point despite pixel noise.
    wsum = sx = sy = 0.0
    weights = []
    for i in range(16):
        for j in range(16):
            w = max(0.0, img[i * 16 + j] - 0.16)
            weights.append(w)
            wsum += w
            sx += w * j
            sy += w * i

    if wsum < 1e-9:
        cx = cy = 7.5
        a, b, th = 3.0, 2.0, 0.0
    else:
        cx, cy = sx / wsum, sy / wsum
        xx = yy = xy = 0.0
        for i in range(16):
            for j in range(16):
                w = weights[i * 16 + j]
                dx, dy = j - cx, i - cy
                xx += w * dx * dx
                yy += w * dy * dy
                xy += w * dx * dy
        xx, yy, xy = xx / wsum, yy / wsum, xy / wsum
        disc = math.sqrt(max(0.0, (xx - yy) ** 2 + 4.0 * xy * xy))
        l1 = max(0.5, (xx + yy + disc) * 0.5)
        l2 = max(0.5, (xx + yy - disc) * 0.5)
        a, b = math.sqrt(4.0 * l1), math.sqrt(4.0 * l2)
        th = 0.5 * math.atan2(2.0 * xy, xx - yy)
        if a < b:
            a, b = b, a
            th += math.pi * 0.5
        a = min(4.8, max(1.2, a))
        b = min(4.8, max(1.2, b))

    def evaluate(params):
        cx, cy, a, b, th = params
        base = _shape(cx, cy, a, b, th, 1.0)
        dot = sum(x * y for x, y in zip(img, base))
        den = sum(x * x for x in base) + 1e-12
        amp = max(0.0, min(1.2, dot / den))
        err = sum((y - amp * x) ** 2 for x, y in zip(base, img))
        return err, amp

    params = [cx, cy, a, b, th]
    err, amp = evaluate(params)

    # Directly refine the parameters against all pixels using the decoder's
    # image model; the amplitude is fitted analytically at each trial.
    steps = [0.8, 0.8, 0.7, 0.7, 0.25]
    for _ in range(12):
        improved = False
        for k in range(5):
            for direction in (-1.0, 1.0):
                trial = params[:]
                trial[k] += direction * steps[k]
                trial[0] = max(0.0, min(15.0, trial[0]))
                trial[1] = max(0.0, min(15.0, trial[1]))
                trial[2] = max(1.1, min(5.0, trial[2]))
                trial[3] = max(1.1, min(5.0, trial[3]))
                e, aa = evaluate(trial)
                if e < err:
                    params, err, amp = trial, e, aa
                    improved = True
        if not improved:
            steps = [s * 0.55 for s in steps]
    return params + [amp]

def decode(z):
    return _shape(z[0], z[1], z[2], z[3], z[4], z[5])
