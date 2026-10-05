def encode(img):
    import math
    
    # Robust initialization using covariance
    sx = sy = sxx = syy = sxy = m = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.15:
                sx += v * j
                sy += v * i
                sxx += v * j * j
                syy += v * i * i
                sxy += v * j * i
                m += v
    
    if m < 0.05:
        z = [8.0, 8.0, 2.5, 2.0, 0.0, 0.7]
    else:
        cx = sx / m
        cy = sy / m
        cxx = sxx / m - cx * cx
        cyy = syy / m - cy * cy
        cxy = sxy / m - cx * cy
        
        trace = cxx + cyy
        det = cxx * cyy - cxy * cxy
        discriminant = max(0, trace * trace / 4 - det)
        lam1 = trace / 2 + math.sqrt(discriminant)
        lam2 = trace / 2 - math.sqrt(discriminant)
        
        a = math.sqrt(max(0.5, lam1)) * 1.6
        b = math.sqrt(max(0.5, lam2)) * 1.6
        th = 0.5 * math.atan2(2 * cxy, cxx - cyy) if abs(cxy) > 1e-6 else 0.0
        amp = max(img)
        z = [cx, cy, a, b, th, amp]
    
    # Gradient descent with adaptive learning rate
    learning_rate = 0.08
    prev_error = float('inf')
    
    for iteration in range(150):
        recon = decode(z)
        error = sum((recon[i] - img[i]) ** 2 for i in range(256))
        
        # Adaptive learning rate
        if error > prev_error:
            learning_rate *= 0.9
        else:
            learning_rate = min(0.15, learning_rate * 1.02)
        prev_error = error
        
        # Gradient via finite differences
        grad = []
        for param_idx in range(6):
            z_plus = z[:]
            z_plus[param_idx] += 0.008
            recon_plus = decode(z_plus)
            error_plus = sum((recon_plus[i] - img[i]) ** 2 for i in range(256))
            grad.append((error_plus - error) / 0.008)
        
        # Update z with momentum-like decay
        for i in range(6):
            z[i] -= learning_rate * grad[i]
        
        # Clamp to reasonable ranges
        z[0] = max(-0.5, min(16.5, z[0]))  # cx
        z[1] = max(-0.5, min(16.5, z[1]))  # cy
        z[2] = max(0.6, min(5.2, z[2]))    # a
        z[3] = max(0.6, min(5.2, z[3]))    # b
        z[5] = max(0.4, min(1.0, z[5]))    # amp
    
    return z

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
            # Smooth symmetric falloff
            p = (1.0 - rho) * (b + 0.3) + 0.4
            img.append(amp * max(0.0, min(1.0, p)))
    return img
