def encode(img):
    import math
    
    # Moment-based initialization
    sx = sy = sxx = syy = sxy = m = 0.0
    bright_vals = []
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
                bright_vals.append(v)
    
    if m < 0.1:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    
    cx = sx / m
    cy = sy / m
    mxx = sxx / m - cx * cx
    myy = syy / m - cy * cy
    mxy = sxy / m - cx * cy
    
    trace = mxx + myy
    det = mxx * myy - mxy * mxy
    discriminant = max(0.0, trace * trace / 4.0 - det)
    lam1 = trace / 2.0 + math.sqrt(discriminant)
    lam2 = trace / 2.0 - math.sqrt(discriminant)
    
    a = max(1.4, min(4.4, 2.3 * math.sqrt(max(lam1, 0.01))))
    b = max(1.4, min(4.4, 2.3 * math.sqrt(max(lam2, 0.01))))
    
    if abs(mxy) > 1e-6:
        th = 0.5 * math.atan2(2.0 * mxy, mxx - myy)
    else:
        th = 0.0
    
    # Better amplitude: median of brightest values to reduce noise sensitivity
    bright_vals.sort(reverse=True)
    n_top = max(3, min(len(bright_vals) // 2, 12))
    amp = sum(bright_vals[:n_top]) / n_top
    amp = min(1.0, max(0.6, amp))
    
    z = [cx, cy, a, b, th, amp]
    best_z = z[:]
    best_error = float('inf')
    
    # Multi-phase refinement with per-parameter adaptive rates
    phase_configs = [(20, 0.25), (16, 0.12), (14, 0.06)]
    
    for phase, (num_iters, base_step) in enumerate(phase_configs):
        for iteration in range(num_iters):
            recon = decode(z)
            error = sum((recon[i] - img[i]) ** 2 for i in range(256))
            
            if error < best_error:
                best_error = error
                best_z = z[:]
            
            # Smoother decay
            step = base_step * (0.6 ** (iteration / (num_iters - 1)))
            
            if step < 0.0003:
                break
            
            improved = False
            for param_idx in range(6):
                z_plus = z[:]
                z_plus[param_idx] += step
                recon_plus = decode(z_plus)
                error_plus = sum((recon_plus[i] - img[i]) ** 2 for i in range(256))
                
                z_minus = z[:]
                z_minus[param_idx] -= step
                recon_minus = decode(z_minus)
                error_minus = sum((recon_minus[i] - img[i]) ** 2 for i in range(256))
                
                # Better momentum scaling based on improvement magnitude
                if error_plus < error - 1e-8 and error_plus <= error_minus:
                    improvement = (error - error_plus) / (error + 1e-6)
                    z[param_idx] += step * min(1.2, 0.7 + improvement)
                    improved = True
                elif error_minus < error - 1e-8:
                    improvement = (error - error_minus) / (error + 1e-6)
                    z[param_idx] -= step * min(1.2, 0.7 + improvement)
                    improved = True
            
            if not improved and iteration > num_iters // 3:
                break
    
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
            # Smooth sigmoid-like fade over ~1 pixel at boundary
            # Steeper near rho=1, smoother falloff beyond
            if rho < 0.85:
                fade = 1.0
            elif rho < 1.15:
                fade = (1.15 - rho) / 0.3
            else:
                fade = max(0.0, (1.25 - rho) * 0.5)
            img.append(amp * max(0.0, fade))
    return img
