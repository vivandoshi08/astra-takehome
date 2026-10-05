def encode(img):
    import math
    
    # Find weighted centroid of bright region
    sx = sy = m = 0.0
    bright = []
    
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.21:  # Optimized threshold
                sx += v * j
                sy += v * i
                m += v
                bright.append(v)
    
    if m < 0.1:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    
    cx = sx / m
    cy = sy / m
    
    # Weighted covariance matrix
    cxx = cyy = cxy = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.21:
                dx = j - cx
                dy = i - cy
                cxx += v * dx * dx
                cyy += v * dy * dy
                cxy += v * dx * dy
    
    cxx /= m
    cyy /= m
    cxy /= m
    
    # Eigenvalue decomposition for semi-axes
    trace = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = max(0.0, trace * trace / 4.0 - det)
    lambda1 = trace / 2.0 + math.sqrt(disc)
    lambda2 = trace / 2.0 - math.sqrt(disc)
    
    # Refined scaling factor (between best approaches)
    a = math.sqrt(max(lambda1, 0.1)) * 2.265
    b = math.sqrt(max(lambda2, 0.1)) * 2.265
    
    if a < b:
        a, b = b, a
    
    a = max(1.4, min(4.4, a))
    b = max(0.8, min(4.4, b))
    
    # Orientation angle
    th = 0.5 * math.atan2(2.0 * cxy, cxx - cyy) if abs(cxy) > 1e-6 else 0.0
    
    # Amplitude from robust percentile with blend
    if bright:
        bright.sort()
        p1 = bright[int(len(bright) * 0.70)]
        p2 = bright[int(len(bright) * 0.72)]
        amp = 0.6 * p1 + 0.4 * p2
        amp = max(0.6, min(1.0, amp))
    else:
        amp = 0.8
    
    return [cx, cy, a, b, th, amp]


def decode(z):
    import math
    
    cx, cy, a, b, th, amp = z
    ct = math.cos(th)
    st = math.sin(th)
    
    img = []
    for i in range(16):
        for j in range(16):
            dx = j - cx
            dy = i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = math.sqrt((u / a) ** 2 + (v / b) ** 2)
            
            # Refined falloff: smoother transition with optimized steepness
            falloff = 1.192
            p = max(0.0, (1.0 - rho / falloff) * (1.0 + 0.12 * (1.0 - rho / falloff)))
            p = max(0.0, min(1.0, p))
            
            img.append(amp * p)
    
    return img
