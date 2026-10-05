def encode(img):
    import math
    
    # Weighted centroid using soft thresholding
    sx = sy = sxx = syy = sxy = m = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            w = max(0, v - 0.20) ** 1.6
            sx += w * j
            sy += w * i
            sxx += w * j * j
            syy += w * i * i
            sxy += w * j * i
            m += w
    
    if m < 0.1:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    
    cx = sx / m
    cy = sy / m
    
    cxx = sxx / m - cx * cx
    cyy = syy / m - cy * cy
    cxy = sxy / m - cx * cy
    
    trace = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = trace * trace / 4.0 - det
    if disc < 0:
        disc = 0
    lam1 = trace / 2.0 + math.sqrt(disc)
    lam2 = trace / 2.0 - math.sqrt(disc)
    
    a = math.sqrt(max(lam1, 0.1)) * 1.88
    b = math.sqrt(max(lam2, 0.1)) * 1.88
    
    if abs(cxy) > 1e-6:
        th = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    else:
        th = 0.0
    
    # Better amplitude: use mean of brightest quartile
    bright = sorted([v for v in img if v > 0.32], reverse=True)
    amp = sum(bright[:max(1, len(bright)//4)]) / max(1, len(bright)//4) if bright else max(img)
    amp = min(1.0, amp * 1.02)
    
    best_z = [cx, cy, a, b, th, amp]
    best_err = float('inf')
    
    # More iterations with finer-grained refinement
    for iteration in range(6):
        step_scale = 0.75 ** iteration
        steps = [-1, -0.5, 0.5, 1]
        
        for idx in range(6):
            base_step = [0.15, 0.15, 0.12, 0.12, 0.14, 0.12][idx]
            step = base_step * step_scale
            
            for delta_mult in steps:
                z_test = best_z[:]
                z_test[idx] += delta_mult * step
                recon = decode(z_test)
                err = sum((img[k] - recon[k]) ** 2 for k in range(256))
                if err < best_err:
                    best_err = err
                    best_z[idx] += delta_mult * step
    
    return best_z

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
            # Fine-tuned sigmoid parameters
            fade = 1.0 / (1.0 + math.exp(6.5 * (rho - 0.868)))
            img.append(amp * fade)
    return img
