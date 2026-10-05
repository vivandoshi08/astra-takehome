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
            d = (1.0 - rho) / max(grad, 1e-10)
            out.append(amp * max(0.0, min(1.0, d + 0.5)))
    return out

def encode(img):
    def moments(t):
        sw = sx = sy = 0.0
        for i in range(16):
            for j in range(16):
                w = max(0.0, img[16*i+j] - t)
                sw += w
                sx += w*j
                sy += w*i
        if sw < 1e-9:
            return [7.5, 7.5, 2.7, 2.0, 0.0]
        cx, cy = sx/sw, sy/sw
        xx = yy = xy = 0.0
        for i in range(16):
            for j in range(16):
                w = max(0.0, img[16*i+j] - t)
                dx, dy = j-cx, i-cy
                xx += w*dx*dx
                yy += w*dy*dy
                xy += w*dx*dy
        xx, yy, xy = xx/sw, yy/sw, xy/sw
        disc = math.sqrt(max(0.0, (xx-yy)**2 + 4*xy*xy))
        a = max(0.9, min(5.8, 2*math.sqrt(max(0.2, (xx+yy+disc)/2))))
        b = max(0.9, min(5.8, 2*math.sqrt(max(0.2, (xx+yy-disc)/2))))
        th = 0.5*math.atan2(2*xy, xx-yy)
        return [cx, cy, a, b, th]

    def clamp(p):
        return [max(-0.5, min(15.5, p[0])),
                max(-0.5, min(15.5, p[1])),
                max(0.9, min(5.8, p[2])),
                max(0.9, min(5.8, p[3])),
                p[4],
                max(0.35, min(1.2, p[5]))]

    def loss(p):
        r = _render(*p)
        return sum((x-y)*(x-y) for x, y in zip(img, r))

    def optimize(start):
        # Nelder-Mead, with parameter-sized simplex steps.
        steps = [0.18, 0.18, 0.18, 0.18, 0.10, 0.025]
        simplex = [clamp(start[:])]
        for k, step in enumerate(steps):
            q = start[:]
            q[k] += step
            simplex.append(clamp(q))
        vals = [loss(p) for p in simplex]

        for _ in range(500):
            order = sorted(range(7), key=lambda i: vals[i])
            simplex = [simplex[i] for i in order]
            vals = [vals[i] for i in order]
            if max(abs(vals[i]-vals[0]) for i in range(1, 7)) < 1e-8:
                break
            c = [sum(simplex[i][k] for i in range(6))/6.0
                 for k in range(6)]
            worst = simplex[6]
            xr = clamp([c[k] + (c[k]-worst[k]) for k in range(6)])
            fr = loss(xr)
            if fr < vals[0]:
                xe = clamp([c[k] + 2*(xr[k]-c[k]) for k in range(6)])
                fe = loss(xe)
                simplex[6], vals[6] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < vals[5]:
                simplex[6], vals[6] = xr, fr
            else:
                if fr < vals[6]:
                    xc = clamp([c[k] + 0.5*(xr[k]-c[k]) for k in range(6)])
                else:
                    xc = clamp([c[k] + 0.5*(worst[k]-c[k]) for k in range(6)])
                fc = loss(xc)
                if fc < min(fr, vals[6]):
                    simplex[6], vals[6] = xc, fc
                else:
                    for i in range(1, 7):
                        simplex[i] = clamp([
                            simplex[0][k] + 0.5*(simplex[i][k]-simplex[0][k])
                            for k in range(6)])
                        vals[i] = loss(simplex[i])
        best = min(range(7), key=lambda i: vals[i])
        return vals[best], simplex[best]

    best_loss, best = float("inf"), None
    for t in (0.08, 0.16, 0.24, 0.32):
        p = moments(t)
        for seed in (p,
                     [p[0], p[1], p[3], p[2], p[4]+math.pi/2],
                     [p[0], p[1], p[2], p[3], p[4]+0.35],
                     [p[0], p[1], p[2], p[3], p[4]-0.35]):
            shape = seed[:]
            sh = _render(*shape, 1.0)
            amp = max(0.35, min(1.2,
                      sum(x*y for x, y in zip(img, sh)) /
                      max(1e-12, sum(y*y for y in sh))))
            value, result = optimize(shape + [amp])
            if value < best_loss:
                best_loss, best = value, result
    return best

def decode(z):
    return _render(*z)
