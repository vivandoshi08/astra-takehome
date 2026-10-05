def encode(img):
    import math
    
    sx = sy = m = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            sx += v * j
            sy += v * i
            m += v
    
    if m < 0.01:
        return [8.0, 8.0, 2.5, 1.5, 0.0, 0.8]
    
    cx = sx / m
    cy = sy / m
    
    mxx = myy = mxy = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            dx = j - cx
            dy = i - cy
            mxx += v * dx * dx
            myy += v * dy * dy
            mxy += v * dx * dy
    
    trace = mxx + myy
    det = mxx * myy - mxy * mxy
    discriminant = max(0.0, trace * trace - 4.0 * det)
    lambda1 = (trace + math.sqrt(discriminant)) / 2.0
    lambda2 = (trace - math.sqrt(discriminant)) / 2.0
    
    a = math.sqrt(max(0.65, lambda1 / m)) if m > 0 else 2.5
    b = math.sqrt(max(0.65, lambda2 / m)) if m > 0 else 1.5
    th = 0.5 * math.atan2(2.0 * mxy, mxx - myy) if abs(mxy) > 1e-6 else 0.0
    
    bright = sorted([v for v in img if v > 0.22], reverse=True)
    if bright:
        amp = sum(bright[:max(1, len(bright)//4)]) / max(1, len(bright)//4)
    else:
        amp = 0.8
    amp = max(0.58, min(1.0, amp))
    
    z = [cx, cy, a, b, th, amp]
    best_z = z[:]
    best_err = float('inf')
    
    for iteration in range(200):
        recon = decode(z)
        err = sum((recon[i] - img[i]) ** 2 for i in range(256))
        
        if err < best_err:
            best_err = err
            best_z = z[:]
        
        progress = iteration / 220.0
        step_size = 0.045 * math.exp(-3.2 * progress * progress)
        
        deltas = [0.115, 0.115, 0.19, 0.19, 0.145, 0.019]
        
        for param_idx in range(6):
            z_plus = z[:]
            z_plus[param_idx] += deltas[param_idx]
            z_minus = z[:]
            z_minus[param_idx] -= deltas[param_idx]
            
            recon_plus = decode(z_plus)
            recon_minus = decode(z_minus)
            
            err_plus = sum((recon_plus[i] - img[i]) ** 2 for i in range(256))
            err_minus = sum((recon_minus[i] - img[i]) ** 2 for i in range(256))
            
            grad = (err_plus - err_minus) / (2.0 * deltas[param_idx])
            z[param_idx] -= step_size * grad
        
        z[0] = max(0.5, min(15.5, z[0]))
        z[1] = max(0.5, min(15.5, z[1]))
        z[2] = max(0.65, min(5.5, z[2]))
        z[3] = max(0.65, min(5.5, z[3]))
        z[5] = max(0.55, min(1.0, z[5]))
    
    return best_z

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
            p = math.exp(-0.5 * (rho - 1.0) ** 2 / 0.32) if rho > 0.05 else 1.0
            img.append(amp * p)
    return img
