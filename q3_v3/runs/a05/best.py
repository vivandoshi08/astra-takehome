def encode(img):
    import math
    
    sx = sy = sxx = syy = sxy = m = 0.0
    bright_vals = []
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.24:
                sx += v * j
                sy += v * i
                sxx += v * j * j
                syy += v * i * i
                sxy += v * j * i
                m += v
                bright_vals.append(v)
    
    if m < 0.1:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    
    cx = sx / m
    cy = sy / m
    
    cxx = sxx / m - cx * cx
    cyy = syy / m - cy * cy
    cxy = sxy / m - cx * cy
    
    trace = cxx + cyy
    det = cxx * cyy - cxy * cxy
    discriminant = max(0, trace * trace / 4 - det)
    lambda1 = trace / 2 + math.sqrt(discriminant)
    lambda2 = trace / 2 - math.sqrt(discriminant)
    
    a = math.sqrt(max(0.5, lambda1)) * 1.612
    b = math.sqrt(max(0.5, lambda2)) * 1.612
    
    if abs(cxy) > 1e-6:
        th = 0.5 * math.atan2(2 * cxy, cxx - cyy)
    else:
        th = 0.0
    
    bright_vals.sort()
    median_bright = bright_vals[len(bright_vals) // 2] if bright_vals else 0.8
    integral_amp = m / (math.pi * a * b * 0.705)
    amp = 0.52 * median_bright + 0.48 * integral_amp
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
            rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
            edge_width = 0.412
            p = 1.0 - rho + edge_width
            img.append(amp * max(0.0, min(1.0, p)))
    return img
