def encode(img):
    import math
    
    sx = sy = sxx = syy = sxy = m = 0.0
    max_val = 0.0
    
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.28:
                sx += v * j
                sy += v * i
                sxx += v * j * j
                syy += v * i * i
                sxy += v * j * i
                m += v
                max_val = max(max_val, v)
    
    if m < 0.1:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    
    cx = sx / m
    cy = sy / m
    amp = max(0.6, min(1.0, max_val))
    
    cxx = sxx / m - cx * cx
    cyy = syy / m - cy * cy
    cxy = sxy / m - cx * cy
    
    trace = cxx + cyy
    det = cxx * cyy - cxy * cxy
    discriminant = max(0.0, trace * trace / 4.0 - det)
    lam1 = trace / 2.0 + math.sqrt(discriminant)
    lam2 = trace / 2.0 - math.sqrt(discriminant)
    
    a = max(1.2, math.sqrt(lam1) * 1.92)
    b = max(1.2, math.sqrt(lam2) * 1.92)
    
    if abs(cxy) > 1e-6:
        th = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    else:
        th = 0.0
    
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
            rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
            # Smooth quadratic fade instead of linear
            p = amp * max(0.0, 1.0 - rho * rho / 1.84)
            img.append(p)
    return img
