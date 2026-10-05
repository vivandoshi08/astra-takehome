import math

def encode(img):
    def decode(z):
        cx, cy, a, b, th, amp = z
        ct, st = math.cos(th), math.sin(th)
        result = []
        for i in range(16):
            for j in range(16):
                dx = j - cx
                dy = i - cy
                u = dx * ct + dy * st
                v = -dx * st + dy * ct
                rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
                p = math.exp(-3.2 * max(0, rho - 0.96) ** 2)
                result.append(amp * p)
        return result
    
    def mse(z):
        recon = decode(z)
        return sum((img[i] - recon[i]) ** 2 for i in range(256)) / 256.0
    
    # Weighted centroid
    sx = sy = m = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.09:
                sx += v * j
                sy += v * i
                m += v
    
    if m == 0.0:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    
    cx, cy = sx / m, sy / m
    
    # Covariance matrix
    u_sum = v_sum = u2_sum = v2_sum = uv_sum = wt = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.09:
                dx = j - cx
                dy = i - cy
                u_sum += v * dx
                v_sum += v * dy
                u2_sum += v * dx * dx
                v2_sum += v * dy * dy
                uv_sum += v * dx * dy
                wt += v
    
    if wt > 0:
        var_u = u2_sum / wt - (u_sum / wt) ** 2
        var_v = v2_sum / wt - (v_sum / wt) ** 2
        cov_uv = uv_sum / wt - (u_sum / wt) * (v_sum / wt)
        
        trace = var_u + var_v
        det = var_u * var_v - cov_uv * cov_uv
        lambda1 = trace / 2.0 + math.sqrt(max(0, trace * trace / 4.0 - det))
        lambda2 = trace / 2.0 - math.sqrt(max(0, trace * trace / 4.0 - det))
        
        a = max(1.4, math.sqrt(max(lambda1, 0.01)) * 2.1)
        b = max(1.4, math.sqrt(max(lambda2, 0.01)) * 2.1)
        
        if abs(var_u - var_v) > 0.01:
            th = 0.5 * math.atan2(2.0 * cov_uv, var_u - var_v)
        else:
            th = 0.0
    else:
        a, b, th = 3.0, 2.0, 0.0
    
    # Amplitude: use high-intensity pixels
    bright_vals = sorted([img[i] for i in range(256) if img[i] > 0.20], reverse=True)
    if len(bright_vals) > 1:
        amp = bright_vals[len(bright_vals) // 4]
    else:
        amp = max(img) if max(img) > 0.1 else 0.8
    amp = max(0.6, min(1.0, amp))
    
    z = [cx, cy, a, b, th, amp]
    best_z = z[:]
    best_mse = mse(z)
    
    step_sizes = [0.25, 0.25, 0.38, 0.38, 0.28, 0.14]
    
    for iteration in range(180):
        improved = False
        
        for param_idx in range(6):
            z_plus = z[:]
            z_plus[param_idx] += step_sizes[param_idx]
            mse_plus = mse(z_plus)
            
            z_minus = z[:]
            z_minus[param_idx] -= step_sizes[param_idx]
            mse_minus = mse(z_minus)
            
            current_mse = mse(z)
            if mse_plus < current_mse - 1e-9:
                z = z_plus
                improved = True
            elif mse_minus < current_mse - 1e-9:
                z = z_minus
                improved = True
        
        current_mse = mse(z)
        if current_mse < best_mse - 1e-10:
            best_mse = current_mse
            best_z = z[:]
        
        if not improved:
            for i in range(6):
                step_sizes[i] *= 0.64
            if max(step_sizes) < 2e-5:
                break
    
    return best_z

def decode(z):
    cx, cy, a, b, th, amp = z
    ct, st = math.cos(th), math.sin(th)
    result = []
    for i in range(16):
        for j in range(16):
            dx = j - cx
            dy = i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
            p = math.exp(-3.2 * max(0, rho - 0.96) ** 2)
            result.append(amp * p)
    return result
