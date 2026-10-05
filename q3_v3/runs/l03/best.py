def encode(img):
    import math

    # Thresholding suppresses most background noise while retaining the ellipse.
    weights = [max(0.0, v - 0.22) for v in img]
    mass = sum(weights)
    if mass < 1e-8:
        return [7.5, 7.5, 2.5, 2.0, 0.0, 0.8]

    cx = sum((k % 16) * w for k, w in enumerate(weights)) / mass
    cy = sum((k // 16) * w for k, w in enumerate(weights)) / mass
    xx = sum(((k % 16) - cx) ** 2 * w for k, w in enumerate(weights)) / mass
    yy = sum(((k // 16) - cy) ** 2 * w for k, w in enumerate(weights)) / mass
    xy = sum(((k % 16) - cx) * ((k // 16) - cy) * w
             for k, w in enumerate(weights)) / mass

    disc = math.sqrt(max(0.0, (xx - yy) ** 2 + 4.0 * xy * xy))
    l1 = max(0.2, (xx + yy + disc) * 0.5)
    l2 = max(0.2, (xx + yy - disc) * 0.5)
    a = max(1.4, min(4.4, 2.0 * math.sqrt(l1)))
    b = max(1.4, min(4.4, 2.0 * math.sqrt(l2)))
    theta = 0.5 * math.atan2(2.0 * xy, xx - yy)

    def render(params):
        x, y, aa, bb, th = params
        ct, st = math.cos(th), math.sin(th)
        out = []
        for i in range(16):
            for j in range(16):
                dx, dy = j - x, i - y
                u = dx * ct + dy * st
                v = -dx * st + dy * ct
                rho = math.sqrt((u / aa) ** 2 + (v / bb) ** 2)
                t = (1.0 - rho) * min(aa, bb) + 0.5
                out.append(max(0.0, min(1.0, t)))
        return out

    def score(params):
        shape = render(params)
        dot = sum(v * q for v, q in zip(img, shape))
        norm = sum(q * q for q in shape)
        amp = max(0.0, min(1.2, dot / max(norm, 1e-12)))
        err = sum((v - amp * q) ** 2 for v, q in zip(img, shape))
        return err, amp

    params = [cx, cy, a, b, theta]
    steps = [0.7, 0.7, 0.55, 0.55, 0.25]
    best, amp = score(params)

    for _ in range(7):
        changed = True
        while changed:
            changed = False
            for d in range(5):
                for direction in (-1.0, 1.0):
                    trial = params[:]
                    trial[d] += direction * steps[d]
                    if d in (2, 3):
                        trial[d] = max(1.3, min(4.8, trial[d]))
                    err, trial_amp = score(trial)
                    if err < best:
                        params, best, amp = trial, err, trial_amp
                        changed = True
        steps = [s * 0.5 for s in steps]

    return params + [amp]


def decode(z):
    import math

    cx, cy, a, b, theta, amp = z
    ct, st = math.cos(theta), math.sin(theta)
    img = []
    for i in range(16):
        for j in range(16):
            dx, dy = j - cx, i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = math.sqrt((u / a) ** 2 + (v / b) ** 2)
            t = (1.0 - rho) * min(a, b) + 0.5
            img.append(amp * max(0.0, min(1.0, t)))
    return img
