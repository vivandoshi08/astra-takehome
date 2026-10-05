import math

def encode(img):
    # Covariance-based initialization
    sx = sy = sxx = syy = sxy = m = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.115:
                sx += v * j
                sy += v * i
                sxx += v * j * j
                syy += v * i * i
                sxy += v * j * i
                m += v
    
    if m < 0.1:
        return [8.0, 8.0, 2.5, 2.0, 0.0, 0.75]
    
    cx = sx / m
    cy = sy / m
    
    cxx = sxx / m - cx * cx
    cyy = syy / m - cy * cy
    cxy = sxy / m - cx * cy
    
    trace = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = max(0, trace * trace / 4.0 - det)
    lam1 = trace / 2.0 + math.sqrt(disc)
    lam2 = trace / 2.0 - math.sqrt(disc)
    
    a = max(0.75, min(6.6, math.sqrt(max(0.1, lam1)) * 1.162))
    b = max(0.75, min(6.6, math.sqrt(max(0.1, lam2)) * 1.162))
    
    th = 0.5 * math.atan2(2.0 * cxy, cxx - cyy) if abs(cxy) > 1e-6 else 0.0
    
    # Robust amplitude from calibrated percentile
    sorted_v = sorted(img, reverse=True)
    amp = sorted_v[max(0, int(len(img) * 0.035))]
    amp = max(0.52, min(1.12, amp))
    
    z = [cx, cy, a, b, th, amp]
    
    # Optimized multi-phase schedule
    phase_schedule = [
        (0.65, 38),
        (0.36, 44),
        (0.19, 50),
        (0.095, 59),
        (0.048, 63),
        (0.018, 56),
        (0.006, 48),
        (0.0018, 35)
    ]
    
    for step_size, num_iters in phase_schedule:
        for iteration in range(num_iters):
            recon = decode(z)
            error = sum((img[i] - recon[i]) ** 2 for i in range(256))
            
            best_z = z[:]
            best_error = error
            
            for dim in range(6):
                for direction in [-1, 1]:
                    z_test = z[:]
                    z_test[dim] += direction * step_size
                    
                    z_test[0] = max(-2.0, min(18.0, z_test[0]))
                    z_test[1] = max(-2.0, min(18.0, z_test[1]))
                    z_test[2] = max(0.70, min(6.6, z_test[2]))
                    z_test[3] = max(0.70, min(6.6, z_test[3]))
                    z_test[4] = z_test[4] % (2 * math.pi)
                    z_test[5] = max(0.40, min(1.20, z_test[5]))
                    
                    recon_test = decode(z_test)
                    error_test = sum((img[i] - recon_test[i]) ** 2 for i in range(256))
                    
                    if error_test < best_error:
                        best_error = error_test
                        best_z = z_test
            
            if best_error < error * 0.999945:
                z = best_z
            else:
                break
    
    return z

def decode(z):
    cx, cy, a, b, th, amp = z
    ct = math.cos(th)
    st = math.sin(th)
    img = []
    for i in range(16):
        for j in range(16):
            dx = j - cx
            dy = i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = math.sqrt((u / a) ** 2 + (v / b) ** 2)
            p = (0.875 - rho) / 0.264
            if p >= 1.0:
                val = amp
            elif p <= 0.0:
                val = 0.0
            else:
                val = amp * p
            img.append(val)
    return img
