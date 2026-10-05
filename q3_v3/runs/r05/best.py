def encode(img):
    import math
    
    # Find centroid of bright region using threshold
    sx = sy = m = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.25:
                sx += v * j
                sy += v * i
                m += v
    
    if m < 0.1:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    
    cx = sx / m
    cy = sy / m
    
    # Compute covariance matrix of bright pixels
    u2 = v2 = uv = 0.0
    for i in range(16):
        for j in range(16):
            vpix = img[i * 16 + j]
            if vpix > 0.25:
                dx = j - cx
                dy = i - cy
                u2 += vpix * dx * dx
                v2 += vpix * dy * dy
                uv += vpix * dx * dy
    
    u2 /= m
    v2 /= m
    uv /= m
    
    # Eigenvalue decomposition
    trace = u2 + v2
    det = u2 * v2 - uv * uv
    disc = max(0, trace * trace / 4 - det)
    lam1 = trace / 2 + math.sqrt(disc)
    lam2 = trace / 2 - math.sqrt(disc)
    
    # Semi-axes: refined scaling accounting for fade
    a = math.sqrt(max(0.2, lam1)) * 2.05
    b = math.sqrt(max(0.2, lam2)) * 2.05
    a, b = max(a, b), min(a, b)
    
    # Orientation
    if abs(u2 - v2) < 0.001 and abs(uv) < 0.001:
        th = 0.0
    else:
        th = 0.5 * math.atan2(2 * uv, u2 - v2)
    
    # Amplitude: use 90th percentile for better robustness
    bright_vals = sorted([img[i * 16 + j] for i in range(16) for j in range(16) if img[i * 16 + j] > 0.25])
    if bright_vals:
        idx = max(0, int(0.90 * len(bright_vals)))
        amp = bright_vals[min(idx, len(bright_vals) - 1)]
    else:
        amp = 0.8
    amp = max(0.6, min(1.0, amp))
    
    return [cx, cy, a, b, th, amp]

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
            
            # Ellipse distance
            rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
            
            # Smooth fade tuned for ~1 pixel rim
            edge = 0.75
            p = (1.0 - rho) / edge
            
            if p >= 1.0:
                pixel = amp
            elif p <= 0.0:
                pixel = 0.0
            else:
                # Smooth cubic Hermite interpolation
                pixel = amp * p * (3.0 - 2.0 * p)
            
            img.append(pixel)
    return img
