def encode(img):
    import math
    
    # Find center via weighted centroid using pixels above threshold
    sx = sy = m = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.24:
                sx += v * j
                sy += v * i
                m += v
    
    if m < 0.1:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    
    cx, cy = sx / m, sy / m
    
    # Amplitude: median of top bright pixels for robustness
    bright = sorted([img[i * 16 + j] for i in range(16) for j in range(16) if img[i * 16 + j] > 0.35], reverse=True)
    amp = bright[min(3, len(bright) - 1)] if bright else max(img)
    amp = max(0.6, min(1.0, amp))
    
    # Compute second moments for ellipse shape
    mxx = myy = mxy = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.24:
                dx = j - cx
                dy = i - cy
                w = v
                mxx += w * dx * dx
                myy += w * dy * dy
                mxy += w * dx * dy
    
    if m > 0:
        mxx /= m
        myy /= m
        mxy /= m
    
    # Eigenanalysis
    trace = mxx + myy
    det = mxx * myy - mxy * mxy
    
    if trace > 0 and det > 0:
        discriminant = trace * trace - 4 * det
        if discriminant >= 0:
            lambda1 = (trace + math.sqrt(discriminant)) / 2.0
            lambda2 = (trace - math.sqrt(discriminant)) / 2.0
            a = max(1.4, min(4.4, math.sqrt(max(lambda1, 0.06)) * 1.337))
            b = max(1.4, min(4.4, math.sqrt(max(lambda2, 0.06)) * 1.337))
            
            if abs(mxy) > 0.01:
                th = 0.5 * math.atan2(2.0 * mxy, mxx - myy)
            else:
                th = 0.0
        else:
            a, b, th = 3.0, 2.0, 0.0
    else:
        a, b, th = 3.0, 2.0, 0.0
    
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
            p = 0.5 * (1.0 + math.tanh(3.12 * (1.105 - rho)))
            img.append(amp * p)
    return img
