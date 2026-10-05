def encode(img):
    import math
    
    sx = sy = sxx = syy = sxy = m = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.25:
                sx += v * j
                sy += v * i
                sxx += v * j * j
                syy += v * i * i
                sxy += v * j * i
                m += v
    
    if m < 0.01:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    
    cx = sx / m
    cy = sy / m
    
    cxx = sxx / m - cx * cx
    cyy = syy / m - cy * cy
    cxy = sxy / m - cx * cy
    
    trace = cxx + cyy
    det = cxx * cyy - cxy * cxy
    discriminant = max(0, trace * trace / 4.0 - det)
    lam1 = trace / 2.0 + math.sqrt(discriminant)
    lam2 = trace / 2.0 - math.sqrt(discriminant)
    
    a = max(0.6, math.sqrt(max(0, lam1)) * 1.82)
    b = max(0.6, math.sqrt(max(0, lam2)) * 1.82)
    
    if abs(cxy) > 1e-6:
        th = 0.5 * math.atan2(2 * cxy, cxx - cyy)
    else:
        th = 0.0
    
    sorted_vals = sorted([img[i * 16 + j] for i in range(16) for j in range(16)], reverse=True)
    amp = sorted_vals[max(0, min(25, len(sorted_vals) - 1))]
    amp = max(0.55, min(1.0, amp))
    
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
            rho = ((u / (a + 0.01)) ** 2 + (v / (b + 0.01)) ** 2) ** 0.5
            
            geom_mean = math.sqrt(a * b + 0.001)
            falloff_width = geom_mean * 0.64
            
            p = (1.0 - rho) * falloff_width + 0.51
            img.append(amp * max(0.0, min(1.0, p)))
    return img
