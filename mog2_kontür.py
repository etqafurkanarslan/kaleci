#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense  ❱  Yeşil Top + Falsolu Çarpma Kestirimi
--------------------------------------------------
• HSV + hareket maskesi  ➜  top merkezi
• Derinlik ➜  3-B (X,Y,Z)
• 2. derece polinom (X(Z), Y(Z))  ➜  duvar düzleminde hit noktası
"""

import cv2, time, numpy as np, pyrealsense2 as rs

# ────────────── RealSense kurulumu ──────────────
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
profile = pipe.start(cfg)
align   = rs.align(rs.stream.color)

intr  = profile.get_stream(rs.stream.color)\
               .as_video_stream_profile().get_intrinsics()
FX, FY, CX, CY = intr.fx, intr.fy, intr.ppx, intr.ppy

# ────────────── Parametreler ──────────────
HSV_LO   = np.array([36,  50,  50], np.uint8)
HSV_HI   = np.array([75, 255, 255], np.uint8)
KERNEL   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
BG_RATE  = 0.02          # arka plan model hızı
HIST_MAX = 8             # son N nokta polinom fit
MIN_PTS  = 5             # en az kaç nokta ile fit
DZ_DRAW  = 0.10          # top duvara 10 cm’den yakınsa ok/çarpı çizme

# ────────────── Değişkenler ──────────────
bg_gray_f = None
hist_xyz  = []           # sadece XYZ saklayacağız
z_wall    = None         # 'w' ile ayarlanır
last_rgb  = -1
prev_t    = time.time()

# ────────────── Yardımcı ──────────────
def pixel_to_cam(u: int, v: int, z: float) -> np.ndarray:
    X = (u - CX) * z / FX
    Y = (v - CY) * z / FY
    return np.array([X, Y, z])

# ────────────── Döngü ──────────────
try:
    while True:
        frames  = align.process(pipe.wait_for_frames())
        color_f = frames.get_color_frame()
        depth_f = frames.get_depth_frame()
        if not color_f or not depth_f:
            continue

        # RGB kuyruk donmasını engelle
        if color_f.get_frame_number() == last_rgb:
            cv2.waitKey(1);   continue
        last_rgb = color_f.get_frame_number()

        color = np.asanyarray(color_f.get_data())
        hsv   = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        # ── hareket maskesi ──
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if bg_gray_f is None:
            bg_gray_f = gray.astype(np.float32);  continue
        cv2.accumulateWeighted(gray, bg_gray_f, BG_RATE)
        diff  = cv2.absdiff(gray, cv2.convertScaleAbs(bg_gray_f))
        _, motion_mask = cv2.threshold(diff, 0, 255,
                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # ── HSV + motion AND ──
        mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, 2)
        mask = cv2.bitwise_and(mask, motion_mask)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c          = max(cnts, key=cv2.contourArea)
            (x, y), r  = cv2.minEnclosingCircle(c.astype(np.float32))
            cx, cy     = int(x), int(y)

            # ── derinlik: 5×5 medyan ──
            xs, ys = np.meshgrid(np.arange(cx-2, cx+3),
                                 np.arange(cy-2, cy+3))
            xs = np.clip(xs, 0, depth_f.get_width()-1).astype(int).ravel()
            ys = np.clip(ys, 0, depth_f.get_height()-1).astype(int).ravel()
            zz = [depth_f.get_distance(ix, iy) for ix, iy in zip(xs, ys)]
            zz = [d for d in zz if d > 0]
            dist = float(np.median(zz)) if zz else 0.0
            if dist == 0:
                continue

            # ── çizimler ──
            cv2.circle(color, (cx, cy), int(r), (0, 255, 255), 2)
            cv2.putText(color, f"{dist:.2f} m", (cx-30, cy-20),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 0), 2)

            # ── 3-B nokta, listeye ekle ──
            xyz = pixel_to_cam(cx, cy, dist)
            hist_xyz.append(xyz)
            hist_xyz = hist_xyz[-HIST_MAX:]         # kuyruk

            # ── çarpma tahmini ──
            if z_wall and len(hist_xyz) >= MIN_PTS:
                P   = np.array(hist_xyz)            # k×3
                Z   = P[:, 2]
                A   = np.column_stack((Z**2, Z, np.ones_like(Z)))

                # X(Z) ve Y(Z) 2. derece fit
                coeff_x = np.linalg.lstsq(A, P[:, 0], rcond=None)[0]
                coeff_y = np.linalg.lstsq(A, P[:, 1], rcond=None)[0]

                dz = z_wall - xyz[2]
                if dz > DZ_DRAW:                    # duvara hâlâ mesafe varsa
                    Z_hit = z_wall
                    X_hit = coeff_x @ np.array([Z_hit**2, Z_hit, 1])
                    Y_hit = coeff_y @ np.array([Z_hit**2, Z_hit, 1])

                    u_hit = int(X_hit * FX / Z_hit + CX)
                    v_hit = int(Y_hit * FY / Z_hit + CY)

                    cv2.drawMarker(color, (u_hit, v_hit), (0, 0, 255),
                                   cv2.MARKER_TILTED_CROSS, 18, 2)
                    cv2.arrowedLine(color, (cx, cy), (u_hit, v_hit),
                                    (0, 0, 255), 2, tipLength=.2)
                    cv2.putText(color, "pred", (u_hit+10, v_hit+10),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)

        # ── FPS ──
        now  = time.time()
        fps  = 1.0 / max(now - prev_t, 1e-6);  prev_t = now
        cv2.putText(color, f"{fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # ── pencereler ──
        cv2.imshow("Mask", mask)
        cv2.imshow("Preview", color)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            # duvara bakarken 'w' ➜ z_wall medyanı al
            h, w = depth_f.get_height(), depth_f.get_width()
            cx_m, cy_m = w//2, h//2
            xs, ys = np.meshgrid(np.arange(cx_m-15, cx_m+16),
                                 np.arange(cy_m-15, cy_m+16))
            zz = [depth_f.get_distance(ix, iy)
                  for ix, iy in zip(xs.ravel(), ys.ravel()) if depth_f.get_distance(ix, iy) > 0]
            if zz:
                z_wall = float(np.median(zz))
                print(f"[INFO] z_wall set to {z_wall:.2f} m")

finally:
    pipe.stop()
    cv2.destroyAllWindows()
