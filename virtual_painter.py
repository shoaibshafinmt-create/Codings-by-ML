"""
Virtual Painter — pure OpenCV edition.

Track a colored object (blue pen cap by default) for drawing, and
optionally a red object on your right finger for colour selection.

Run:      python virtual_painter.py
Controls: hover the tracked object over the top toolbar to pick a colour/eraser
          OR hover a RED object (right finger) over the toolbar to pick a colour
          c = clear canvas | s = save PNG | t = toggle HSV tuner | q = quit

Author: Muhammad Hamzah
"""

import time

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CAM_W, CAM_H = 640, 480
TOOLBAR_H = 60
BRUSH = 8
ERASER = 40

PALETTE = [
    {"name": "RED",    "bgr": (0, 0, 255)},
    {"name": "GREEN",  "bgr": (0, 255, 0)},
    {"name": "BLUE",   "bgr": (255, 0, 0)},
    {"name": "YELLOW", "bgr": (0, 255, 255)},
    {"name": "PURPLE", "bgr": (255, 0, 255)},
    {"name": "ERASE",  "bgr": (40, 40, 40)},
]
SWATCH_W = CAM_W // len(PALETTE)

# Default HSV range for a BLUE object (pen cap, marker lid, bottle cap...)
# Press 't' at runtime to open the tuner and adjust for your object/lighting.
HSV_LOWER = np.array([100, 120, 60])
HSV_UPPER = np.array([130, 255, 255])

# ---------------------------------------------------------------------------
# Added: HSV ranges for a RED object on your right finger
# ---------------------------------------------------------------------------
# Red wraps around 0 in OpenCV's HSV, so we need two masks.
RED_LOWER1 = np.array([0, 120, 60])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 120, 60])
RED_UPPER2 = np.array([180, 255, 255])

MIN_AREA = 400  # ignore tiny specks of matching color (noise)


# --------------------------------------------------------------------------- #
# HSV tuner (optional, toggled with 't')
# --------------------------------------------------------------------------- #
def _noop(_):
    pass


def open_tuner():
    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Tuner", 400, 300)
    for name, val, mx in [
        ("H low", HSV_LOWER[0], 179), ("S low", HSV_LOWER[1], 255), ("V low", HSV_LOWER[2], 255),
        ("H high", HSV_UPPER[0], 179), ("S high", HSV_UPPER[1], 255), ("V high", HSV_UPPER[2], 255),
    ]:
        cv2.createTrackbar(name, "HSV Tuner", int(val), mx, _noop)


def read_tuner():
    lo = np.array([
        cv2.getTrackbarPos("H low", "HSV Tuner"),
        cv2.getTrackbarPos("S low", "HSV Tuner"),
        cv2.getTrackbarPos("V low", "HSV Tuner"),
    ])
    hi = np.array([
        cv2.getTrackbarPos("H high", "HSV Tuner"),
        cv2.getTrackbarPos("S high", "HSV Tuner"),
        cv2.getTrackbarPos("V high", "HSV Tuner"),
    ])
    return lo, hi


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def draw_toolbar(frame, active_idx):
    for i, item in enumerate(PALETTE):
        x0, x1 = i * SWATCH_W, (i + 1) * SWATCH_W
        cv2.rectangle(frame, (x0, 0), (x1, TOOLBAR_H), item["bgr"], -1)
        if item["name"] == "ERASE":
            cv2.putText(frame, "ERASE", (x0 + 6, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if i == active_idx:
            cv2.rectangle(frame, (x0, 0), (x1, TOOLBAR_H), (255, 255, 255), 4)
    return frame


def find_pointer(mask):
    """Return (x, y) of the tracked object's tip, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None
    # Use the topmost point of the contour as the "pen tip"
    tip = tuple(c[c[:, :, 1].argmin()][0])
    return tip


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    global HSV_LOWER, HSV_UPPER

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Is another app using it?")
        return

    canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    prev_point = None
    active_idx = 0
    tuner_open = False

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    print("Virtual Painter running.")
    print("Track a BLUE object to draw. Track a RED object on your right finger to change colour.")
    print("Keys: c=clear  s=save  t=tuner  q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Failed to read frame from webcam.")
            break

        frame = cv2.resize(frame, (CAM_W, CAM_H))
        frame = cv2.flip(frame, 1)  # mirror

        if tuner_open:
            HSV_LOWER, HSV_UPPER = read_tuner()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ---- Blue object (drawing) -------------------------------------------------
        mask_blue = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_DILATE, kernel)

        tip_blue = find_pointer(mask_blue)

        # ---- Red object (colour selection) -----------------------------------------
        mask_red1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        mask_red2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_DILATE, kernel)

        tip_red = find_pointer(mask_red)

        # ---- Colour selection from toolbar -----------------------------------------
        # Priority: if a red finger tip is in the toolbar, it changes the colour.
        # Otherwise the blue drawing tip can also change it (original behaviour).
        if tip_red is not None:
            rx, ry = tip_red
            if ry < TOOLBAR_H:
                active_idx = min(rx // SWATCH_W, len(PALETTE) - 1)
            # Draw a small red indicator for the selection finger
            cv2.circle(frame, (rx, ry), 10, (0, 0, 255), 2)
        elif tip_blue is not None:
            x, y = tip_blue
            if y < TOOLBAR_H:
                active_idx = min(x // SWATCH_W, len(PALETTE) - 1)

        # ---- Drawing with the blue tip ---------------------------------------------
        if tip_blue is not None:
            x, y = tip_blue
            # If the blue tip is in the toolbar and no red tip was overriding,
            # we already handled selection above. Still draw the indicator.
            if y >= TOOLBAR_H:
                is_eraser = PALETTE[active_idx]["name"] == "ERASE"
                color = (0, 0, 0) if is_eraser else PALETTE[active_idx]["bgr"]
                thickness = ERASER if is_eraser else BRUSH

                if prev_point is None:
                    prev_point = (x, y)
                cv2.line(canvas, prev_point, (x, y), color, thickness)
                prev_point = (x, y)

            # Indicator for drawing tip
            ind_color = (255, 255, 255) if PALETTE[active_idx]["name"] == "ERASE" \
                else PALETTE[active_idx]["bgr"]
            cv2.circle(frame, (x, y), 10, ind_color, 2)
        else:
            prev_point = None

        # Composite canvas over live frame
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        composed = cv2.add(frame_bg, canvas)

        composed = draw_toolbar(composed, active_idx)
        cv2.putText(composed, "c:clear  s:save  t:tuner  q:quit",
                    (10, CAM_H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        cv2.imshow("Virtual Painter", composed)
        if tuner_open:
            # Show both masks for debugging
            combined_mask = cv2.bitwise_or(mask_blue, mask_red)
            cv2.imshow("Mask (what the tracker sees)", combined_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas[:] = 0
        elif key == ord("s"):
            fname = f"painting_{int(time.time())}.png"
            cv2.imwrite(fname, canvas)
            print(f"Saved {fname}")
        elif key == ord("t"):
            if not tuner_open:
                open_tuner()
                tuner_open = True
            else:
                cv2.destroyWindow("HSV Tuner")
                cv2.destroyWindow("Mask (what the tracker sees)")
                tuner_open = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()