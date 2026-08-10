"""
Demonstrates uniform quantization of a grayscale image to L gray levels,
using the formula from part (a):

    q(r) = floor(r / delta) * delta + delta/2      where delta = 256 / L

Shows quantized images for L = 2..256 in a matplotlib grid (false contouring
visible at small L), and plots MSQE vs. L.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


# (i) Vectorized quantization function (no per-pixel Python loops)
def quantize(img, L):
    """
    Uniformly quantize an 8-bit grayscale image to L gray levels.

    q(r) = floor(r / delta) * delta + delta/2,   delta = 256 / L

    Fully vectorized with NumPy. Works on the whole image array at once.

    Parameters
    ----------
    img : np.ndarray, dtype=uint8
        Input grayscale image, values in [0, 255].
    L : int
        Number of quantization levels (e.g. 2, 4, 8, ...).

    Returns
    -------
    np.ndarray, dtype=uint8
        Quantized image, same shape as img.
    """
    img_f = img.astype(np.float64)  # avoid uint8 overflow during math
    delta = 256.0 / L

    bin_index = np.floor(img_f / delta)  # floor(r / delta), vectorized
    q = bin_index * delta + delta / 2.0  # midpoint of each bin

    q = np.round(q)  # round after all calculations
    q = np.clip(q, 0, 255)  # safety clip (guards float edge cases)

    return q.astype(np.uint8)


# (ii) Read a grayscale image and produce quantized versions
image_path = "images/image1.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

L_values = [2, 4, 8, 16, 32, 64, 128, 256]
quantized_images = {L: quantize(img, L) for L in L_values}

# (iii) Display all versions in a matplotlib grid
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for ax, L in zip(axes, L_values):
    ax.imshow(quantized_images[L], cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"L = {L}")
    ax.axis("off")

fig.suptitle(
    "Uniform Quantization at Different L (false contouring visible for small L)"
)
plt.tight_layout()
plt.savefig("images/quantized_grid.png", dpi=150)
plt.show()


# (iv) MSQE between original image and each quantized version, plot MSQE vs L
def msqe(original, quantized):
    """Mean Squared Quantization Error between two same-shape uint8 images."""
    orig_f = original.astype(np.float64)
    quant_f = quantized.astype(np.float64)
    error = orig_f - quant_f
    return np.mean(error**2)


msqe_values = [msqe(img, quantized_images[L]) for L in L_values]

print("\nMSQE per level:")
for L, e in zip(L_values, msqe_values):
    print(f"  L = {L:>3}  ->  MSQE = {e:.3f}")

plt.figure(figsize=(7, 5))
plt.plot(L_values, msqe_values, marker="o")
plt.xscale("log", base=2)
plt.xticks(L_values, [str(L) for L in L_values])
plt.xlabel("Number of quantization levels (L)")
plt.ylabel("MSQE")
plt.title("MSQE vs. L")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("images/msqe_vs_L.png", dpi=150)
plt.show()

# (v) Discussion: trend of the plot, relation to part (a) numerical result
print("DISCUSSION")
print("""
    The MSQE-vs-L plot decreases sharply and monotonically as L increases. This
    follows directly from the formula: delta = 256/L, so larger L means a
    smaller step size delta, meaning each pixel is quantized to a value much
    closer to its true intensity. Since MSQE is dominated by delta^2-scale
    rounding error, doubling L roughly quarters the MSQE (MSQE decreases
    approximately as 1/L^2) -- the curve drops steeply between L=2 and L=16,
    then flattens out and approaches 0 as L approaches 256 (no quantization
    loss at all, since 256 levels exactly matches 8-bit precision).

    This matches the hand-computed result from part (a): for the 8 sample
    pixel values quantized to L = 4, the MSQE was 433.0, a large error caused
    by the coarse 64-wide bins. In this program's plot, the L = 4 point on the
    curve should sit at a similarly high MSQE value near the steep, high-error
    end of the curve, while L = 256 sits at (or near) MSQE = 0. The visual
    counterpart of this numerical trend is that images quantized with small L
    (e.g. L = 2, 4) show strong false contouring / posterization -- visible
    banding where smooth gradients used to be -- while by L = 32 or higher the
    image is visually almost indistinguishable from the original.
""")
