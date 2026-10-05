def encode(img):
    sx = sy = m = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.3:
                sx += v * j; sy += v * i; m += v
    if m == 0:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    r = (m / (0.8 * 3.141592653589793)) ** 0.5
    return [sx / m, sy / m, r, r, 0.0, 0.8]

def decode(z):
    import math
    cx, cy, a, b, th, amp = z
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
