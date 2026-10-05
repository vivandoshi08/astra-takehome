def encode(img):
    import math
    
    sx = sy = m = 0.0
    max_val = 0.0
    bright_pixels = []
    
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.27:
                sx += v * j
                sy += v * i
                m += v
                max_val = max(max_val, v)
                bright_pixels.append(v)
    
    if m < 0.1:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.5]
    
    cx, cy = sx / m, sy / m
    
    # Amplitude: slightly increased max_val weight from 0.63
    bright_pixels.sort(reverse=True)
    n_top = max(1, len(bright_pixels) // 3)
    top_avg = sum(bright_pixels[:n_top]) / n_top
    amp = 0.62 * max_val + 0.38 * top_avg
    
    cov_xx = cov_yy = cov_xy = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.27:
                dx, dy = j - cx, i - cy
                cov_xx += v * dx * dx
                cov_yy += v * dy * dy
                cov_xy += v * dx * dy
    
    cov_xx /= m
    cov_yy /= m
    cov_xy /= m
    
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    discriminant = max(0.0, trace * trace - 4 * det)
    lambda1 = (trace + discriminant ** 0.5) / 2
    lambda2 = (trace - discriminant ** 0.5) / 2
    
    # Refined semi-axes calibration
    a = max(lambda1, lambda2) ** 0.5 * 1.86
    b = min(lambda1, lambda2) ** 0.5 * 1.86
    
    a = max(1.2, min(5.0, a))
    b = max(0.8, min(4.5, b))
    
    th = 0.0
    if abs(cov_xy) > 1e-7 or abs(cov_xx - cov_yy) > 1e-7:
        th = 0.5 * math.atan2(2 * cov_xy, cov_xx - cov_yy)
    
    return [cx, cy, a, b, th, amp]


def decode(z):
    import math
    cx, cy, a, b, th, amp = z
    ct, st = math.cos(th), math.sin(th)
    img = []
    for i in range(16):
        for j in range(16):
            dx, dy = j - cx, i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
            # Refined edge fade tuning for optimal reconstruction
            edge_width = max(0.22, min(a, b) * 0.117)
            fade_start = 1.0 - edge_width * 0.135
            if rho < fade_start:
                p = 1.0
            else:
                p = max(0.0, 1.0 - (rho - fade_start) / edge_width)
            img.append(amp * p)
    return img
