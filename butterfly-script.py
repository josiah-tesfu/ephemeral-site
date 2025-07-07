# Phase 1: Preprocessing
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# = == Configurable Parameters ===
gamma = 1.5                 # Gamma correction factor
brightness_threshold = 20   # Brightness floor threshold (0–255)
alpha_sobel = 0     # Strength of soft enhancement
canny_boost = 255     # Brightness to add at hard edges


# === Load and Convert to Grayscale ===
input_path = "butterfly-hand-combined.jpg"
img = Image.open(input_path).convert("RGB")
gray = ImageOps.grayscale(img)
gray_np = np.array(gray).astype(np.float32)

# === Apply Gamma Correction ===
gamma_corrected = 255 * ((gray_np / 255) ** gamma)

# === Apply Brightness Threshold Floor ===
gamma_corrected[gamma_corrected < brightness_threshold] = 0
gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

import cv2

# === Convert gamma-corrected image to uint8 for OpenCV ===
gamma_uint8 = gamma_corrected.astype(np.uint8)

# === Compute Sobel Edge Map ===
sobel_x = cv2.Sobel(gamma_uint8, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gamma_uint8, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.hypot(sobel_x, sobel_y)
sobel_mag = np.clip(sobel_mag / sobel_mag.max(), 0, 1) * 255
sobel_mag = sobel_mag.astype(np.uint8)

# === Compute Canny Edge Map ===

canny_edges = cv2.Canny(gamma_uint8, threshold1=50, threshold2=150)
canny_mask = (canny_edges > 0).astype(np.uint8)

# === Blend Both into Brightness Map ===

enhanced = gamma_uint8 + (alpha_sobel * sobel_mag).astype(np.uint8)
enhanced = np.clip(enhanced, 0, 255)
enhanced += (canny_mask * canny_boost)
enhanced = np.clip(enhanced, 0, 255)

# === Apply Brightness Threshold Floor (Post-Blend) ===
enhanced[enhanced < brightness_threshold] = 0
enhanced = enhanced.astype(np.uint8)

# === Save and Display ===
final_img = Image.fromarray(enhanced)
final_img.save("01_preprocessed2.png")

plt.imshow(final_img, cmap="gray")
plt.axis("off")
plt.title("01_preprocessed2.png (hybrid edge enhanced)")
plt.show()

import random
from PIL import ImageDraw, Image
import numpy as np
import matplotlib.pyplot as plt

# === Phase 2: Dot Sampling ===

# === Configurable Parameters ===
num_dots_target = 2500        # Total number of dots to sample
dot_radius = 4                # Radius of each dot in pixels
overlap_allowed = 0.3         # e.g., allow 50% diameter overlap
delta = 1                   # Controls brightness influence (delta < 1 favors dark, >1 favors light)

# === Load Preprocessed Image ===
gray_img = Image.open("01_preprocessed2.png")
gray_np = np.array(gray_img).astype(np.float32)
height, width = gray_np.shape

# === Normalize to Probability Map and Apply Delta Curve ===
prob_map = (gray_np / 255.0) ** delta

# === Sample Dot Coordinates ===
coords = []
attempts = 0
max_attempts = num_dots_target * 10

while len(coords) < num_dots_target and attempts < max_attempts:
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)
    if random.random() < prob_map[y, x]:
        coords.append((x, y))
    attempts += 1

# === Filter Overlapping Dots ===
def filter_overlapping(coords, dot_radius, overlap_allowed):
    min_dist = 2 * dot_radius * (1 - overlap_allowed)
    min_dist_sq = min_dist ** 2
    accepted = []

    for x, y in coords:
        too_close = False
        for ax, ay in accepted:
            dx = x - ax
            dy = y - ay
            if dx * dx + dy * dy < min_dist_sq:
                too_close = True
                break
        if not too_close:
            accepted.append((x, y))
    return accepted

# === Apply Overlap Constraint ===
coords = filter_overlapping(coords, dot_radius, overlap_allowed)
print(f"Total dots after overlap filtering: {len(coords)}")

# === Render Dots ===
dot_image = Image.new("RGB", (width, height), "black")
draw = ImageDraw.Draw(dot_image)

for x, y in coords:
    draw.ellipse(
        [(x - dot_radius, y - dot_radius), (x + dot_radius, y + dot_radius)],
        fill="white"
    )

# === Save and Preview ===
dot_image.save("02_white_dots2.png")
plt.imshow(dot_image)
plt.axis("off")
plt.title("02_white_dots2.png")
plt.show()

import colorsys

# === Configurable Parameters ===
light_thresh = 200
mid_thresh = 0
chroma_threshold = 0.1  # Below this, favor more neutral color in bin

# === Color Palette (bin-sorted) ===
color_bins = {
    'light': {
        'colors': {
            'Pale Ivory': '#F2E8D5',
            'Sky Blue': '#B8D9F9',
        }
    },
    'mid': {
        'colors': {
            'Warm Orange': '#F6A06D',
            'Lavender': '#9E85C4',
            'Muted Green': '#6BBF59',
        }
    },
    'dark': {
        'colors': {
            'Warm Brown': '#523F3A',
            'Slate Blue': '#303952',
        }
    }
}

# Convert hex to RGB (0–255) and to HSL (0–1)
def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hsl(rgb):
    r, g, b = [v / 255.0 for v in rgb]
    return colorsys.rgb_to_hls(r, g, b)  # Returns (hue, lightness, saturation)

for bin_data in color_bins.values():
    color_data = []
    for name, hex_val in bin_data['colors'].items():
        rgb = hex_to_rgb(hex_val)
        h, l, s = rgb_to_hsl(rgb)
        chroma = s
        color_data.append({
            'name': name,
            'hex': hex_val,
            'rgb': rgb,
            'hue': h,
            'chroma': chroma
        })
    bin_data['colors'] = color_data

# === Load dot coordinates and original image ===
original_rgb = Image.open(input_path).convert("RGB")
dot_image = Image.new("RGB", original_rgb.size, "black")
draw = ImageDraw.Draw(dot_image)

colored_coords = []

for x, y in coords:
    x_i, y_i = int(round(x)), int(round(y))
    if not (0 <= x_i < original_rgb.width and 0 <= y_i < original_rgb.height):
        continue

    r, g, b = original_rgb.getpixel((x_i, y_i))
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Assign tone bin
    if lum >= light_thresh:
        tone_bin = 'light'
    elif lum >= mid_thresh:
        tone_bin = 'mid'
    else:
        tone_bin = 'dark'

    h, l, s = rgb_to_hsl((r, g, b))
    chroma = s

    # Pick best match in bin
    best_color = None
    min_hue_diff = float('inf')
    for c in color_bins[tone_bin]['colors']:
        if chroma < chroma_threshold:
            best_color = min(color_bins[tone_bin]['colors'], key=lambda cc: cc['chroma'])
            break
        hue_diff = abs(h - c['hue'])
        if hue_diff > 0.5:
            hue_diff = 1.0 - hue_diff  # Wrap-around
        if hue_diff < min_hue_diff:
            min_hue_diff = hue_diff
            best_color = c

    draw.ellipse(
        [(x - dot_radius, y - dot_radius), (x + dot_radius, y + dot_radius)],
        fill=best_color['hex']
    )
    colored_coords.append((x, y, best_color['hex']))

# === Phase 4: Accent Injection ===
# Assumes: `colored_coords` from Phase 3 and `original_rgb` already loaded

# === Tunable Parameters ===
accent_chroma_threshold = 0.4
accent_luminance_min = 60
accent_luminance_max = 230
red_hue_range = [(0.95, 1.0), (0.0, 0.05)]
yellow_hue_range = [(0.12, 0.18)]

# === Accent Colors ===
accent_colors = {
    'Bright Red': '#FA5252',
    'Vivid Yellow': '#FEE440'
}

# === Hue Matching Function ===
def in_hue_range(h, ranges):
    for low, high in ranges:
        if low <= h <= high:
            return True
    return False

# === Process All Dots ===
updated_coords = []

for x, y, prev_color in colored_coords:
    x_i, y_i = int(round(x)), int(round(y))
    if not (0 <= x_i < original_rgb.width and 0 <= y_i < original_rgb.height):
        updated_coords.append((x, y, prev_color))
        continue

    r, g, b = original_rgb.getpixel((x_i, y_i))
    h, l, s = rgb_to_hsl((r, g, b))
    chroma = s
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    if (chroma > accent_chroma_threshold and 
        accent_luminance_min <= luminance <= accent_luminance_max):
        
        if in_hue_range(h, red_hue_range):
            updated_coords.append((x, y, accent_colors['Bright Red']))
            continue
        elif in_hue_range(h, yellow_hue_range):
            updated_coords.append((x, y, accent_colors['Vivid Yellow']))
            continue

    updated_coords.append((x, y, prev_color))

# === Re-render Image with Accents ===
accent_image = Image.new("RGB", original_rgb.size, "black")
draw = ImageDraw.Draw(accent_image)

for x, y, color in updated_coords:
    draw.ellipse(
        [(x - dot_radius, y - dot_radius), (x + dot_radius, y + dot_radius)],
        fill=color
    )   

accent_image.save("04_colored_with_accents2.png")
plt.imshow(accent_image)
plt.axis("off")
plt.title("04_colored_with_accents2.png")
plt.show()

from collections import Counter

# === Count and Print Final Color Usage ===
color_counts = Counter(color for _, _, color in updated_coords)
total_dots = sum(color_counts.values())

# Map hex back to color names
hex_to_name = {v: k for k, v in {
    **{c['name']: c['hex'] for b in color_bins.values() for c in b['colors']},
    **accent_colors
}.items()}

# Print counts and percentages
print("\n=== Dot Counts by Color ===")
for hex_val, count in color_counts.most_common():
    name = hex_to_name.get(hex_val, hex_val)
    pct = 100 * count / total_dots
    print(f"{name:<15} {count:>5}  ({pct:5.2f}%)")


import os
import json
import re
import random
from PIL import Image, ImageDraw

# === Configuration ===
input_path = "butterfly-hand-combined.jpg"
coords_js_path = "updated_coords.js"
butterfly_folder = "butterfly_colored"
output_path = "butterfly_render.png"
dot_radius = 6.4  # scale reference
dot_diameter = dot_radius * 2

# === Load Coordinates from JS ===
with open(coords_js_path, "r") as f:
    js_content = f.read()
match = re.search(r"const updated_coords\s*=\s*(\[[\s\S]*?\]);", js_content)
if not match:
    raise ValueError("updated_coords array not found in JS file.")
array_str = match.group(1).replace("'", '"')
updated_coords = json.loads(array_str)

# === Load Base Image ===
base_img = Image.open(input_path).convert("RGBA")

# === Prepare Composite Image ===
composite = Image.new("RGBA", base_img.size, (0, 0, 0, 255))

# === Place Butterflies ===
for x, y, color in updated_coords:
    hex_color = color.lower().lstrip('#')
    
    # Map color hex to image filename
    filename_map = {
        'f2e8d5': 'pale_ivory',
        'b8d9f9': 'sky_blue',
        'f6a06d': 'warm_orange',
        '9e85c4': 'lavender',
        '6bbf59': 'muted_green',
        '523f3a': 'warm_brown',
        '303952': 'slate_blue',
        'fa5252': 'bright_red',
        'fee440': 'vivid_yellow',
    }
    
    if hex_color not in filename_map:
        continue

    butterfly_path = os.path.join(butterfly_folder, f"butterfly_{filename_map[hex_color]}.png")
    if not os.path.exists(butterfly_path):
        continue

    # Load and transform butterfly image
    butterfly = Image.open(butterfly_path).convert("RGBA")
    scale = dot_diameter / max(butterfly.size)  # scale to fit within dot
    new_size = (int(butterfly.width * scale), int(butterfly.height * scale))
    butterfly_resized = butterfly.resize(new_size, resample=Image.LANCZOS)

    # Apply random rotation
    angle = random.uniform(0, 360)
    butterfly_rotated = butterfly_resized.rotate(angle, expand=True)

    # Compute paste coordinates
    paste_x = int(x - butterfly_rotated.width // 2)
    paste_y = int(y - butterfly_rotated.height // 2)

    # Paste with transparency
    composite.alpha_composite(butterfly_rotated, dest=(paste_x, paste_y))

# === Save Final Output ===
composite.save(output_path)

# Display result
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.imshow(composite)
plt.axis("off")
plt.title("Butterfly Render")
plt.show()


# PHASE 4 #

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import json
import re

# === Load updated_coords from JS file ===
with open("updated_coords.js", "r") as f:
    js_content = f.read()

match = re.search(r"const updated_coords\s*=\s*(\[[\s\S]*?\]);", js_content)
if not match:
    raise ValueError("updated_coords array not found in JS file.")

array_str = match.group(1).replace("'", '"')
updated_coords = json.loads(array_str)

# === Load Original Image ===
input_path = "butterfly-hand-combined.jpg"
original_rgb = Image.open(input_path).convert("RGB")

# === Color to Number Mapping ===
color_number_map = {
    '#F2E8D5': '1',  # Pale Ivory
    '#B8D9F9': '2',  # Sky Blue
    '#F6A06D': '3',  # Warm Orange
    '#9E85C4': '4',  # Lavender
    '#6BBF59': '5',  # Muted Green
    '#523F3A': '6',  # Warm Brown
    '#303952': '7',  # Slate Blue
    '#FA5252': '8',  # Bright Red
    '#FEE440': '9',  # Vivid Yellow
}

# === Create Projection Image with Numbers ===
proj_image = Image.new("RGB", original_rgb.size, "black")
draw = ImageDraw.Draw(proj_image)

try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=12)
except:
    font = ImageFont.load_default()

for x, y, color in updated_coords:
    label = color_number_map.get(color.upper(), '?')
    draw.text((x, y), label, fill="white", anchor="mm", font=font)

proj_image.save("04_color_numbers_projection.png")
plt.imshow(proj_image)
plt.axis("off")
plt.title("04_color_numbers_projection.png")
plt.show()

# PHASE 5 #

from PIL import Image, ImageDraw, ImageFont
import os
import json
import re

# === Configurable Parameters ===
input_path = "butterfly-hand-combined.jpg"
coords_js_path = "updated_coords.js"
output_dir = "grid_projections"
font_size = 12
grid_size = 5
padding_px = 20         # padding around crop
border_px = 10          # uniform black border
bounding_box_color = "white"
crosshair_size = 10

# === Load updated_coords from JS file ===
with open(coords_js_path, "r") as f:
    js_content = f.read()
match = re.search(r"const updated_coords\s*=\s*(\[[\s\S]*?\]);", js_content)
if not match:
    raise ValueError("updated_coords array not found.")
array_str = match.group(1).replace("'", '"')
updated_coords = json.loads(array_str)

# === Color to Number Mapping ===
color_number_map = {
    '#F2E8D5': '1', '#B8D9F9': '2', '#F6A06D': '3',
    '#9E85C4': '4', '#6BBF59': '5', '#523F3A': '6',
    '#303952': '7', '#FA5252': '8', '#FEE440': '9',
}

# === Load Base Image ===
base_img = Image.open(input_path).convert("RGB")
width, height = base_img.size
assert width == height, "Image must be square."

cell_size = width // grid_size
proj_cell_size = cell_size + 2 * padding_px
final_size = proj_cell_size + 2 * border_px  # add border on all sides

# === Prepare Font ===
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
except:
    font = ImageFont.load_default()

# === Make Output Directory ===
os.makedirs(output_dir, exist_ok=True)

# === Generate Each Grid Cell ===
for row in range(grid_size):
    for col in range(grid_size):
        x0 = col * cell_size - padding_px
        y0 = row * cell_size - padding_px
        x1 = x0 + proj_cell_size
        y1 = y0 + proj_cell_size

        proj_img = Image.new("RGB", (final_size, final_size), "black")
        draw = ImageDraw.Draw(proj_img)

        for x, y, color in updated_coords:
            if x0 <= x < x1 and y0 <= y < y1:
                label = color_number_map.get(color.upper(), '?')
                x_draw = x - x0 + border_px
                y_draw = y - y0 + border_px
                draw.text((x_draw, y_draw), label, fill="white", anchor="mm", font=font)

        # Draw bounding box
        bbox_start = border_px
        bbox_end = final_size - border_px - 1
        draw.rectangle([bbox_start, bbox_start, bbox_end, bbox_end], outline=bounding_box_color, width=1)

        # Draw center crosshairs
        center_x = final_size // 2
        center_y = final_size // 2
        cross = crosshair_size // 2
        draw.line((center_x - cross, center_y, center_x + cross, center_y), fill="white")
        draw.line((center_x, center_y - cross, center_x, center_y + cross), fill="white")

        filename = f"grid_{row}_{col}.png"
        proj_img.save(os.path.join(output_dir, filename))

# PHASE 6 #

from PIL import Image, ImageDraw, ImageFont

# Settings
text = "EPHEMERAL"
font_path = "Forum-Regular.ttf"
font_size = 200
image_size = (1024, 165)
output_path = "ephemeral_text.png"

# Create image
img = Image.new("L", image_size, 0)  # Black background
draw = ImageDraw.Draw(img)
font = ImageFont.truetype(font_path, font_size)

# Get bounding box
bbox = font.getbbox(text)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2 - 35)

# Draw text with faux bold effect
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        draw.text((position[0] + dx, position[1] + dy), text, fill=255, font=font)

# Save
img.save(output_path)

import random
from PIL import ImageDraw, Image
import numpy as np
import matplotlib.pyplot as plt

# === Configurable Parameters ===
num_dots_target = 700
dot_radius = 3
overlap_allowed = 0.5
delta = 1

# === Load Image ===
gray_img = Image.open("ephemeral_text.png")
gray_np = np.array(gray_img).astype(np.float32)
height, width = gray_np.shape

# === Probability Map ===
prob_map = (gray_np / 255.0) ** delta

# === Sample Coordinates ===
ephemeral_coords = []
attempts = 0
max_attempts = num_dots_target * 10

while len(ephemeral_coords) < num_dots_target and attempts < max_attempts:
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)
    if random.random() < prob_map[y, x]:
        ephemeral_coords.append((x, y))
    attempts += 1

# === Filter Overlaps ===
def filter_overlapping(coords, dot_radius, overlap_allowed):
    min_dist = 2 * dot_radius * (1 - overlap_allowed)
    min_dist_sq = min_dist ** 2
    accepted = []
    for x, y in coords:
        too_close = False
        for ax, ay in accepted:
            dx = x - ax
            dy = y - ay
            if dx * dx + dy * dy < min_dist_sq:
                too_close = True
                break
        if not too_close:
            accepted.append((x, y))
    return accepted

ephemeral_coords = filter_overlapping(ephemeral_coords, dot_radius, overlap_allowed)
print(f"Total dots after overlap filtering: {len(ephemeral_coords)}")

# === Save JS File in [x, y, "#FFFFFF"] format ===
with open("ephemeral_coords.js", "w") as f:
    f.write("var ephemeral_coords = [\n")
    for x, y in ephemeral_coords:
        f.write(f"  [{x}, {y}, \"#FFFFFF\"],\n")
    f.write("];\n")

# === Render Image ===
dot_image = Image.new("RGB", (width, height), "black")
draw = ImageDraw.Draw(dot_image)

for x, y in ephemeral_coords:
    draw.ellipse(
        [(x - dot_radius, y - dot_radius), (x + dot_radius, y + dot_radius)],
        fill="white"
    )

dot_image.save("ephemeral-dots.png")
plt.imshow(dot_image)
plt.axis("off")
plt.title("ephemeral-dots.png")
plt.show()
