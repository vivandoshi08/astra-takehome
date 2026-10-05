def encode(img):
    import math
    
    max_val = max(img)
    threshold = max(0.13, max_val * 0.31)
    
    # Weighted centroid
    sx = sy = m = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > threshold:
                sx += v * j
                sy += v * i
                m += v
    
    if m < 0.05:
        return [8.0, 8.0, 2.5, 2.0, 0.0, 0.8]
    
    cx = sx / m
    cy = sy / m
    
    # Second moments for shape
    sxx = syy = sxy = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > threshold * 0.66:
                dx = j - cx
                dy = i - cy
                sxx += v * dx * dx
                syy += v * dy * dy
                sxy += v * dx * dy
    
    # Eigenvalues
    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    discriminant = max(0, trace * trace - 4 * det)
    lambda1 = (trace + math.sqrt(discriminant)) / 2.0
    lambda2 = (trace - math.sqrt(discriminant)) / 2.0
    
    scale = 2.115
    a = max(0.7, math.sqrt(lambda1 / m) * scale)
    b = max(0.7, math.sqrt(lambda2 / m) * scale)
    
    th = 0.0 if abs(sxy) < 1e-6 else 0.5 * math.atan2(2 * sxy, sxx - syy)
    
    amp = min(1.0, max(0.6, max_val))
    
    return [cx, cy, a, b, th, amp]

def decode(z):
    import math
    cx, cy, a, b, th, amp = z
    ct, st = math.cos(th), math.sin(th)
    img = []
    fade_width = 0.88
    
    for i in range(16):
        for j in range(16):
            dx = j - cx
            dy = i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
            
            edge_dist = rho - (1.0 - fade_width)
            
            if edge_dist < 0:
                p = 1.0
            elif edge_dist > fade_width:
                p = 0.0
            else:
                p = 0.5 * (1.0 + math.cos(math.pi * edge_dist / fade_width))
            
            img.append(amp * p)
    return img
