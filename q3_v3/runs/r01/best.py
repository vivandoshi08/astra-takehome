def encode(img):
    import math
    
    # Find amplitude and weighted centroid
    amp = 0.0
    sx = sy = m = 0.0
    threshold = 0.32
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            amp = max(amp, v)
            if v > threshold:
                sx += v * j
                sy += v * i
                m += v
    
    if m < 0.1:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    
    cx = sx / m
    cy = sy / m
    
    # Covariance matrix of bright pixels
    mxx = myy = mxy = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > threshold:
                dx = j - cx
                dy = i - cy
                mxx += v * dx * dx
                myy += v * dy * dy
                mxy += v * dx * dy
    
    mxx /= m
    myy /= m
    mxy /= m
    
    # Eigenvalue decomposition
    trace = mxx + myy
    det = mxx * myy - mxy * mxy
    disc = max(0, trace * trace * 0.25 - det)
    lam1 = trace * 0.5 + math.sqrt(disc)
    lam2 = trace * 0.5 - math.sqrt(disc)
    
    # Semi-axes with refined scaling
    a = math.sqrt(max(0.1, lam1 * 3.92))
    b = math.sqrt(max(0.1, lam2 * 3.92))
    
    if a < b:
        a, b = b, a
    
    # Orientation
    if abs(mxy) > 1e-6:
        theta = 0.5 * math.atan2(2 * mxy, mxx - myy)
    else:
        theta = 0.0
    
    amp = max(0.6, min(1.0, amp))
    
    return [cx, cy, a, b, theta, amp]

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
            # Smoothed edge transition with soft clipping
            p = 1.0 - rho
            # Apply smooth fade over ~1 pixel accounting for rim
            fade = p * (1.0 + 0.5 * b * (1.0 - max(0.0, min(1.0, p))))
            val = max(0.0, min(1.0, fade + 0.42))
            img.append(amp * val)
    return img
