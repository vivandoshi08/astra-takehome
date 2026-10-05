def encode(img):
    import math
    
    sx = sy = m = 0.0
    sxx = syy = sxy = 0.0
    peak = 0.0
    bright_vals = []
    
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.2:
                sx += v * j
                sy += v * i
                m += v
                peak = max(peak, v)
                if v > 0.55:
                    bright_vals.append(v)
    
    if m < 0.1:
        return [8.0, 8.0, 2.5, 1.5, 0.0, 0.7]
    
    cx, cy = sx / m, sy / m
    
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.2:
                dx = j - cx
                dy = i - cy
                sxx += v * dx * dx
                syy += v * dy * dy
                sxy += v * dx * dy
    
    sxx /= m
    syy /= m
    sxy /= m
    
    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = max(0.0, trace * trace / 4.0 - det)
    
    lam1 = trace / 2.0 + math.sqrt(disc)
    lam2 = trace / 2.0 - math.sqrt(disc)
    
    a = math.sqrt(max(lam1, 0.5)) * 1.5
    b = math.sqrt(max(lam2, 0.5)) * 1.5
    
    th = 0.0
    if abs(sxy) > 1e-6:
        th = 0.5 * math.atan2(2 * sxy, sxx - syy)
    
    if bright_vals:
        bright_vals.sort()
        amp_init = bright_vals[len(bright_vals) // 2]
    else:
        amp_init = peak
    
    best_mse = float('inf')
    best_amp = amp_init
    ct, st = math.cos(th), math.sin(th)
    fade = 0.70
    
    # Coarse search
    for amp_try in [amp_init * f for f in [0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55]]:
        amp_try = min(max(amp_try, 0.35), 1.7)
        mse = 0.0
        for idx in range(256):
            i = idx // 16
            j = idx % 16
            dx = j - cx
            dy = i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
            p = (1.0 - rho) / fade + 0.5
            recon = amp_try * max(0.0, min(1.0, p))
            mse += (recon - img[idx]) ** 2
        if mse < best_mse:
            best_mse = mse
            best_amp = amp_try
    
    # Fine refinement with adaptive shrinking
    for iteration in range(7):
        delta = 0.01 / (1.6 ** iteration)
        for amp_try in [best_amp - 3*delta, best_amp - 2*delta, best_amp - delta, best_amp, best_amp + delta, best_amp + 2*delta, best_amp + 3*delta]:
            amp_try = min(max(amp_try, 0.35), 1.7)
            mse = 0.0
            for idx in range(256):
                i = idx // 16
                j = idx % 16
                dx = j - cx
                dy = i - cy
                u = dx * ct + dy * st
                v = -dx * st + dy * ct
                rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
                p = (1.0 - rho) / fade + 0.5
                recon = amp_try * max(0.0, min(1.0, p))
                mse += (recon - img[idx]) ** 2
            if mse < best_mse:
                best_mse = mse
                best_amp = amp_try
    
    return [cx, cy, a, b, th, best_amp]

def decode(z):
    import math
    cx, cy, a, b, th, amp = z
    ct, st = math.cos(th), math.sin(th)
    fade = 0.70
    img = []
    for i in range(16):
        for j in range(16):
            dx = j - cx
            dy = i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
            p = (1.0 - rho) / fade + 0.5
            img.append(amp * max(0.0, min(1.0, p)))
    return img
