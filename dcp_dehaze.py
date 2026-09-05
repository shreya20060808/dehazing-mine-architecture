import cv2
import numpy as np
import os

# ============================================================
# SETTINGS
# ============================================================

INPUT_IMAGE = "mining_truck.png"

# True = artificially create a hazy image from your clear image
# False = use your original image directly as the hazy input
USE_SYNTHETIC_HAZE = False

OUTPUT_DIR = "dcp_output"

PATCH_SIZE = 15
OMEGA = 0.85
T0 = 0.15


# ============================================================
# DARK CHANNEL
# ============================================================

def dark_channel(image, patch_size=15):

    min_channel = np.min(image, axis=2)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (patch_size, patch_size)
    )

    dark = cv2.erode(min_channel, kernel)

    return dark


# ============================================================
# ATMOSPHERIC LIGHT
# ============================================================

def estimate_atmospheric_light(image, dark):

    h, w = dark.shape

    # Number of brightest pixels used for estimation
    num_pixels = max(int(h * w * 0.001), 1)

    dark_flat = dark.reshape(-1)

    # brightest dark-channel pixels
    indices = np.argsort(dark_flat)[-num_pixels:]

    image_flat = image.reshape(-1, 3)

    atmospheric_candidates = image_flat[indices]

    # Select candidate with maximum RGB intensity
    brightness = np.sum(atmospheric_candidates, axis=1)

    A = atmospheric_candidates[np.argmax(brightness)]

    return A


# ============================================================
# TRANSMISSION ESTIMATION
# ============================================================

def estimate_transmission(image, A, patch_size=15, omega=0.95):

    normalized = image / (A + 1e-6)

    transmission = 1 - omega * dark_channel(
        normalized,
        patch_size
    )

    return transmission


# ============================================================
# GUIDED FILTER
# ============================================================

def guided_filter(guide, src, radius=40, eps=1e-3):

    guide = guide.astype(np.float32)
    src = src.astype(np.float32)

    mean_I = cv2.boxFilter(
        guide,
        cv2.CV_32F,
        (radius, radius)
    )

    mean_p = cv2.boxFilter(
        src,
        cv2.CV_32F,
        (radius, radius)
    )

    mean_Ip = cv2.boxFilter(
        guide * src,
        cv2.CV_32F,
        (radius, radius)
    )

    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(
        guide * guide,
        cv2.CV_32F,
        (radius, radius)
    )

    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)

    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(
        a,
        cv2.CV_32F,
        (radius, radius)
    )

    mean_b = cv2.boxFilter(
        b,
        cv2.CV_32F,
        (radius, radius)
    )

    output = mean_a * guide + mean_b

    return output


# ============================================================
# SCENE RECOVERY
# ============================================================

def recover_scene(image, transmission, A, t0=0.1):

    transmission = np.maximum(
        transmission,
        t0
    )

    recovered = (
        (image - A) / transmission[:, :, None]
    ) + A

    recovered = np.clip(
        recovered,
        0,
        1
    )

    return recovered


# ============================================================
# SYNTHETIC HAZE
# ============================================================

def add_haze(image, atmospheric_light=(0.85, 0.88, 0.92),
             haze_strength=0.72):

    h, w, _ = image.shape

    # Depth-like gradient
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)

    X, Y = np.meshgrid(x, y)

    # More haze toward distance/top
    depth = 0.25 + 0.75 * (1 - Y)

    transmission = np.exp(
        -haze_strength * depth
    )

    A = np.array(
        atmospheric_light,
        dtype=np.float32
    )

    hazy = (
        image * transmission[:, :, None]
        + A * (1 - transmission[:, :, None])
    )

    return np.clip(hazy, 0, 1)


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    original = cv2.imread(
        INPUT_IMAGE
    )

    if original is None:

        print("ERROR: Could not find:")
        print(INPUT_IMAGE)
        print()
        print("Put your image in the same folder as this script.")

        return

    original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    original = original.astype(
        np.float32
    ) / 255.0

    # --------------------------------------------------------
    # CREATE HAZY INPUT
    # --------------------------------------------------------

    if USE_SYNTHETIC_HAZE:

        hazy = add_haze(
            original,
            haze_strength=0.72
        )

    else:

        hazy = original.copy()

    # --------------------------------------------------------
    # DARK CHANNEL
    # --------------------------------------------------------

    dark = dark_channel(
        hazy,
        PATCH_SIZE
    )

    # --------------------------------------------------------
    # ATMOSPHERIC LIGHT
    # --------------------------------------------------------

    A = estimate_atmospheric_light(
        hazy,
        dark
    )

    print()
    print("===================================")
    print("DARK CHANNEL PRIOR")
    print("===================================")

    print(
        "Estimated Atmospheric Light:"
    )

    print(
        "R = {:.3f}".format(A[0])
    )

    print(
        "G = {:.3f}".format(A[1])
    )

    print(
        "B = {:.3f}".format(A[2])
    )

    # --------------------------------------------------------
    # TRANSMISSION MAP
    # --------------------------------------------------------

    transmission = estimate_transmission(
        hazy,
        A,
        PATCH_SIZE,
        OMEGA
    )

    # --------------------------------------------------------
    # REFINE TRANSMISSION
    # --------------------------------------------------------

    guide = cv2.cvtColor(
        (hazy * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0

    transmission_refined = guided_filter(
        guide,
        transmission,
        radius=40,
        eps=1e-3
    )

    transmission_refined = np.clip(
        transmission_refined,
        0,
        1
    )

    # --------------------------------------------------------
    # SCENE RECOVERY
    # --------------------------------------------------------

    dehazed = recover_scene(
        hazy,
        transmission_refined,
        A,
        T0
    )

    # --------------------------------------------------------
    # ATMOSPHERIC LIGHT VISUALIZATION
    #
    # IMPORTANT:
    # A itself is a global RGB estimate.
    # This visualization keeps the truck faintly visible
    # for demonstration purposes.
    # --------------------------------------------------------

    A_image = np.ones_like(hazy) * A

    atmospheric_visualization = (
        0.72 * A_image
        + 0.28 * hazy
    )

    atmospheric_visualization = np.clip(
        atmospheric_visualization,
        0,
        1
    )

    # --------------------------------------------------------
    # CONVERT TO OPENCV BGR
    # --------------------------------------------------------

    hazy_bgr = cv2.cvtColor(
        (hazy * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )

    dark_img = (
        dark * 255
    ).astype(np.uint8)

    transmission_img = (
        transmission_refined * 255
    ).astype(np.uint8)

    atmospheric_bgr = cv2.cvtColor(
        (atmospheric_visualization * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )

    dehazed_bgr = cv2.cvtColor(
        (dehazed * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # SAVE INDIVIDUAL STAGES
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "01_hazy_input.png"
        ),
        hazy_bgr
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "02_dark_channel.png"
        ),
        dark_img
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "03_transmission_map.png"
        ),
        transmission_img
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "04_atmospheric_light.png"
        ),
        atmospheric_bgr
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "05_dehazed_output.png"
        ),
        dehazed_bgr
    )

    # ========================================================
    # CREATE FLOW IMAGE
    # ========================================================

    target_w = 300
    target_h = 230

    stages = [
        ("1. HAZY INPUT", hazy_bgr),
        ("2. DARK CHANNEL", dark_img),
        ("3. TRANSMISSION MAP", transmission_img),
        ("4. ATMOSPHERIC LIGHT", atmospheric_bgr),
        ("5. DEHAZED OUTPUT", dehazed_bgr)
    ]

    # Convert grayscale stages to BGR
    prepared = []

    for title, image in stages:

        if len(image.shape) == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        image = cv2.resize(
            image,
            (target_w, target_h)
        )

        prepared.append(
            (title, image)
        )

    margin = 25
    title_h = 70
    label_h = 45

    canvas_w = (
        margin * 2
        + target_w * 5
        + 20 * 4
    )

    canvas_h = (
        title_h
        + target_h
        + label_h
        + 80
    )

    canvas = np.ones(
        (canvas_h, canvas_w, 3),
        dtype=np.uint8
    ) * 245

    # Title

    cv2.putText(
        canvas,
        "DARK CHANNEL PRIOR (DCP) DEHAZING",
        (margin, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (30, 60, 100),
        2,
        cv2.LINE_AA
    )

    for i, (title, image) in enumerate(prepared):

        x = (
            margin
            + i * (target_w + 20)
        )

        y = title_h

        # Image
        canvas[
            y:y+target_h,
            x:x+target_w
        ] = image

        # Border
        cv2.rectangle(
            canvas,
            (x, y),
            (x+target_w, y+target_h),
            (60, 60, 60),
            2
        )

        # Stage label
        cv2.putText(
            canvas,
            title,
            (x, y+target_h+28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (30, 30, 30),
            1,
            cv2.LINE_AA
        )

        # Arrow
        if i < 4:

            arrow_x1 = x + target_w + 2
            arrow_x2 = x + target_w + 17

            arrow_y = (
                y
                + target_h // 2
            )

            cv2.arrowedLine(
                canvas,
                (arrow_x1, arrow_y),
                (arrow_x2, arrow_y),
                (40, 100, 180),
                2,
                tipLength=0.4
            )

    # Bottom explanation

    text = (
        "Hazy Input → Dark Channel → Transmission Map "
        "→ Atmospheric Light → Scene Recovery"
    )

    cv2.putText(
        canvas,
        text,
        (margin, canvas_h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (40, 80, 120),
        1,
        cv2.LINE_AA
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "dcp_demo.jpg"
        ),
        canvas
    )

    print()
    print("===================================")
    print("DONE!")
    print("===================================")
    print(
        "Results saved in:",
        OUTPUT_DIR
    )
    print()
    print(
        "Open:"
    )
    print(
        os.path.join(
            OUTPUT_DIR,
            "dcp_demo.png"
        )
    )


if __name__ == "__main__":
    main()