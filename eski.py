#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basitleştirilmiş top takibi
• HSV ile konum
• Aynı pikselde derinlik
IR fark / arka-plan adımları kaldırıldı
"""
import cv2, time, numpy as np, pyrealsense2 as rs

# ----------------- RealSense -----------------
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color,    848, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth,    848, 480, rs.format.z16, 90)
cfg.enable_stream(rs.stream.infrared, 1,   848, 480, rs.format.y8, 90)  # opsiyonel
profile  = pipe.start(cfg)
align    = rs.align(rs.stream.color)

# ----------------- Parametreler ---------------
hsv_lo = np.array([36, 50, 50],  dtype=np.uint8)   # yeşil aralık
hsv_hi = np.array([75,255,255],  dtype=np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
prev_t = time.time(); last_rgb = -1

try:
    while True:
        frames = align.process(pipe.wait_for_frames())
        color_f = frames.get_color_frame()
        if not color_f: continue

        # --- donma koruması ---
        if color_f.get_frame_number() == last_rgb:
            cv2.waitKey(1); continue
        last_rgb = color_f.get_frame_number()

        depth_f = frames.get_depth_frame()
        color   = np.asanyarray(color_f.get_data())
        hsv     = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        # -------- HSV maske + kontur ----------
        mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 2)
        cnts,_= cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

        if cnts:
            c   = max(cnts, key=cv2.contourArea)
            (x,y), r = cv2.minEnclosingCircle(c.astype(np.float32))
            cx, cy   = int(x), int(y)

            # ------ derinlik: 5×5 komşuluk ------
            xs, ys = np.meshgrid(np.arange(cx-2,cx+3),
                                 np.arange(cy-2,cy+3))
            xs = np.clip(xs,0,depth_f.get_width()-1).astype(int).flatten()
            ys = np.clip(ys,0,depth_f.get_height()-1).astype(int).flatten()
            zz = [depth_f.get_distance(ix,iy) for ix,iy in zip(xs,ys)]
            zz = [d for d in zz if d>0]
            dist = float(np.median(zz)) if zz else 0.0

            # --------- çizimler -------------
            cv2.circle(color, (cx,cy), int(r), (0,255,255), 2)    # sarı daire
            cv2.putText(color, f"{dist:.2f} m", (cx-30, cy-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # ------------- fps -----------------
        now = time.time(); fps = 1.0 / max(now-prev_t,1e-6); prev_t = now
        cv2.putText(color, f"RGB: {fps:.1f} FPS", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

        # ------------- göster --------------
        cv2.imshow("HSV ROI", mask)
        cv2.imshow("Preview", color)
        if cv2.waitKey(1)&0xFF==ord('q'): break

finally:
    pipe.stop(); cv2.destroyAllWindows()
