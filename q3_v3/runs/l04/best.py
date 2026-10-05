def encode(img):
    import math

    def render(p):
        cx, cy, a, b, th = p
        ct, st = math.cos(th), math.sin(th)
        out = []
        for i in range(16):
            for j in range(16):
                dx, dy = j - cx, i - cy
                u, v = dx * ct + dy * st, -dx * st + dy * ct
                rho = math.sqrt((u / a) ** 2 + (v / b) ** 2)
                if rho < 1e-12:
                    radius = min(a, b)
                else:
                    radius = rho / math.sqrt((u / (a * a)) ** 2 +
                                             (v / (b * b)) ** 2)
                out.append(max(0.0, min(1.0, 0.5 + (1.0 - rho) * radius)))
        return out

    def fit_amp(q):
        dot = norm = 0.0
        for y, x in zip(img, q):
            dot += y * x
            norm += x * x
        amp = max(0.35, min(1.15, dot / max(norm, 1e-12)))
        err = 0.0
        for y, x in zip(img, q):
            d = y - amp * x
            err += d * d
        return err, amp

    def wrap(th):
        return (th + math.pi / 2.0) % math.pi - math.pi / 2.0

    # Thresholded moments at several cutoffs reduce sensitivity to background noise.
    starts = []
    for cutoff in (0.20, 0.32, 0.44):
        sw = sx = sy = 0.0
        for i in range(16):
            for j in range(16):
                w = max(0.0, img[i * 16 + j] - cutoff)
                sw += w
                sx += w * j
                sy += w * i
        if sw <= 1e-9:
            continue
        cx, cy = sx / sw, sy / sw
        xx = yy = xy = 0.0
        for i in range(16):
            for j in range(16):
                w = max(0.0, img[i * 16 + j] - cutoff)
                dx, dy = j - cx, i - cy
                xx += w * dx * dx
                yy += w * dy * dy
                xy += w * dx * dy
        xx, yy, xy = xx / sw, yy / sw, xy / sw
        angle = 0.5 * math.atan2(2.0 * xy, xx - yy)
        disc = math.sqrt(max(0.0, (xx - yy) ** 2 + 4.0 * xy * xy))
        a = max(1.2, min(5.2, 2.0 * math.sqrt(max(0.2, 0.5 * (xx + yy + disc)))))
        b = max(1.2, min(5.2, 2.0 * math.sqrt(max(0.2, 0.5 * (xx + yy - disc)))))
        starts.append([cx, cy, a, b, angle])

    if not starts:
        starts = [[7.5, 7.5, 2.8, 2.0, 0.0]]

    bounds = [(-1.0, 16.0), (-1.0, 16.0),
              (1.2, 5.2), (1.2, 5.2), (-math.pi / 2.0, math.pi / 2.0)]

    def optimize(p):
        p = p[:]
        steps = [0.45, 0.45, 0.45, 0.45, 0.18]
        best, amp = fit_amp(render(p))
        while max(steps) > 0.009:
            changed = False
            for k in range(5):
                old = p[k]
                local_best, local_val, local_amp = best, old, amp
                for sign in (-1.0, 1.0):
                    q = p[:]
                    q[k] = old + sign * steps[k]
                    if k == 4:
                        q[k] = wrap(q[k])
                    else:
                        lo, hi = bounds[k]
                        q[k] = max(lo, min(hi, q[k]))
                    score, candidate_amp = fit_amp(render(q))
                    if score < local_best:
                        local_best, local_val, local_amp = score, q[k], candidate_amp
                if local_val != old:
                    p[k] = local_val
                    best, amp = local_best, local_amp
                    changed = True
            if not changed:
                steps = [s * 0.55 for s in steps]
        return best, p, amp

    # Refine each moment estimate and retain the best pixelwise fit.
    best, bp, ba = float("inf"), None, 0.8
    for p in starts:
        score, q, amp = optimize(p)
        if score < best:
            best, bp, ba = score, q, amp

    # A few nearby starts help escape small moment/angle initialization errors.
    base = bp[:]
    for da, db, dt in ((0.35, -0.35, 0.0), (-0.35, 0.35, 0.0),
                       (0.0, 0.0, 0.22), (0.0, 0.0, -0.22)):
        p = base[:]
        p[2] = max(1.2, min(5.2, p[2] + da))
        p[3] = max(1.2, min(5.2, p[3] + db))
        p[4] = wrap(p[4] + dt)
        score, q, amp = optimize(p)
        if score < best:
            best, bp, ba = score, q, amp

    return bp + [ba]


def decode(z):
    import math

    cx, cy, a, b, th, amp = z
    ct, st = math.cos(th), math.sin(th)
    out = []
    for i in range(16):
        for j in range(16):
            dx, dy = j - cx, i - cy
            u, v = dx * ct + dy * st, -dx * st + dy * ct
            rho = math.sqrt((u / a) ** 2 + (v / b) ** 2)
            if rho < 1e-12:
                radius = min(a, b)
            else:
                radius = rho / math.sqrt((u / (a * a)) ** 2 +
                                         (v / (b * b)) ** 2)
            out.append(amp * max(0.0, min(1.0, 0.5 + (1.0 - rho) * radius)))
    return out
