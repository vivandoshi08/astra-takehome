import math

def encode(img):
    # Compute weighted centroid of bright pixels
    sx = sy = m = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.25:
                sx += v * j
                sy += v * i
                m += v
    
    if m == 0.0:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    
    cx = sx / m
    cy = sy / m
    
    # Covariance matrix of bright pixels for shape
    cxx = cyy = cxy = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.25:
                dx = j - cx
                dy = i - cy
                cxx += v * dx * dx
                cyy += v * dy * dy
                cxy += v * dx * dy
    
    if m > 0:
        cxx /= m
        cyy /= m
        cxy /= m
    
    # Eigenvalue decomposition
    trace = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = max(0.0, trace * trace / 4.0 - det)
    disc = math.sqrt(disc)
    
    lambda1 = trace / 2.0 + disc
    lambda2 = trace / 2.0 - disc
    lambda1 = max(0.01, lambda1)
    lambda2 = max(0.01, lambda2)
    
    a = math.sqrt(lambda1) * 1.86
    b = math.sqrt(lambda2) * 1.86
    
    # Orientation from eigenvector
    if abs(cxy) > 1e-6:
        theta = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    else:
        theta = 0.0
    
    # Amplitude: weighted average of bright pixels with adaptive threshold
    amp_sum = 0.0
    amp_count = 0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.32:
                amp_sum += v
                amp_count += 1
    
    if amp_count > 0:
        amp = amp_sum / amp_count
    else:
        amp = max((img[i * 16 + j] for i in range(16) for j in range(16) if img[i * 16 + j] > 0.25), default=0.8)
    
    return [cx, cy, a, b, theta, amp]

def decode(z):
    cx, cy, a, b, th, amp = z
    ct, st = math.cos(th), math.sin(th)
    img = []
    for i in range(16):
        for j in range(16):
            dx = j - cx
            dy = i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            
            # Ellipse distance with smooth fade
            rho = math.sqrt((u / max(0.1, a)) ** 2 + (v / max(0.1, b)) ** 2)
            
            # Optimized smooth fade with smoother transition
            if rho < 0.64:
                p = 1.0
            elif rho < 1.36:
                # Smooth cubic fade for better gradient
                t = (1.36 - rho) / 0.72
                p = t * t * (3.0 - 2.0 * t)  # Smoothstep
            else:
                p = 0.0
            
            img.append(amp * p)
    return img
