"""Two realistic, longer candidate programs of the kind an LLM writes for this
task (used for the cost estimate and as 'better than seed' candidates)."""

MOMENTS = '''\
import math

def encode(img):
    # Threshold the image, then use first and second image moments of the
    # bright region to estimate centre, semi-axes and orientation.
    thr = 0.3
    m = sx = sy = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > thr:
                m += v
                sx += v * j
                sy += v * i
    if m <= 0.0:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    cx, cy = sx / m, sy / m
    sxx = syy = sxy = 0.0
    inside = []
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > thr:
                dx, dy = j - cx, i - cy
                sxx += v * dx * dx
                syy += v * dy * dy
                sxy += v * dx * dy
                inside.append(v)
    sxx /= m
    syy /= m
    sxy /= m
    # Eigen-decomposition of the 2x2 covariance matrix.
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(tr * tr / 4.0 - det, 0.0))
    l1 = tr / 2.0 + disc
    l2 = max(tr / 2.0 - disc, 1e-6)
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    # For a uniform ellipse, variance along an axis is (semi-axis^2) / 4.
    a = max(2.0 * math.sqrt(l1), 1.0)
    b = max(2.0 * math.sqrt(l2), 1.0)
    inside.sort()
    core = inside[len(inside) // 4:] or inside
    amp = sum(core) / len(core)
    return [cx, cy, a, b, theta, amp]

def decode(z):
    import math
    cx, cy, a, b, th, amp = z
    a = max(a, 0.5)
    b = max(b, 0.5)
    ct, st = math.cos(th), math.sin(th)
    img = []
    for i in range(16):
        for j in range(16):
            dx = j - cx
            dy = i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
            p = (1.0 - rho) * b + 0.5
            img.append(amp * (0.0 if p < 0.0 else (1.0 if p > 1.0 else p)))
    return img
'''

REFINED = '''\
import math

def _render(z):
    cx, cy, a, b, th, amp = z
    a = max(a, 0.5)
    b = max(b, 0.5)
    ct, st = math.cos(th), math.sin(th)
    out = []
    for i in range(16):
        for j in range(16):
            dx = j - cx
            dy = i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = math.sqrt((u / a) ** 2 + (v / b) ** 2)
            p = (1.0 - rho) * b + 0.5
            out.append(amp * (0.0 if p < 0.0 else (1.0 if p > 1.0 else p)))
    return out

def _sse(z, img):
    r = _render(z)
    return sum((x - y) * (x - y) for x, y in zip(r, img))

def _moments(img):
    thr = 0.3
    m = sx = sy = 0.0
    for k, v in enumerate(img):
        if v > thr:
            m += v
            sx += v * (k % 16)
            sy += v * (k // 16)
    if m <= 0.0:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    cx, cy = sx / m, sy / m
    sxx = syy = sxy = 0.0
    vals = []
    for k, v in enumerate(img):
        if v > thr:
            dx, dy = (k % 16) - cx, (k // 16) - cy
            sxx += v * dx * dx
            syy += v * dy * dy
            sxy += v * dx * dy
            vals.append(v)
    sxx, syy, sxy = sxx / m, syy / m, sxy / m
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    disc = math.sqrt(max(tr * tr / 4.0 - det, 0.0))
    l1, l2 = tr / 2.0 + disc, max(tr / 2.0 - disc, 1e-6)
    th = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    vals.sort()
    core = vals[len(vals) // 4:] or vals
    return [cx, cy, max(2.0 * math.sqrt(l1), 1.0),
            max(2.0 * math.sqrt(l2), 1.0), th, sum(core) / len(core)]

def encode(img):
    # Start from moment estimates, then refine all six parameters by
    # coordinate descent on the squared reconstruction error.
    z = _moments(img)
    best = _sse(z, img)
    steps = [0.5, 0.5, 0.5, 0.5, 0.2, 0.05]
    for _ in range(4):
        improved = False
        for k in range(6):
            for sgn in (1.0, -1.0):
                trial = list(z)
                trial[k] += sgn * steps[k]
                e = _sse(trial, img)
                if e < best:
                    best, z, improved = e, trial, True
        if not improved:
            steps = [s * 0.5 for s in steps]
    return z

def decode(z):
    return _render(z)
'''
