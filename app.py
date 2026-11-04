import json
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb, rgb2hsv, hsv2rgb

st.set_page_config(page_title="VTuber Palette Generator", page_icon="🎨", layout="wide")
st.title("🎨 Auto-Dominant Color Palette & Theme Generator")
st.caption("K-Means Clustering pada piksel avatar VTuber (PNG) → tema stream overlay / UI light-dark")

# ---------------- Sidebar ----------------
st.sidebar.header("Pengaturan")
K = st.sidebar.slider("Jumlah kluster (K)", 3, 8, 5)
space = st.sidebar.radio("Ruang warna clustering", ["LAB", "HSV"])
median_size = st.sidebar.select_slider("Ukuran Median Filter", [1, 3, 5, 7], value=3)
alpha_thr = st.sidebar.slider("Ambang alpha (buang piksel transparan)", 1, 255, 128)
max_side = st.sidebar.slider("Resize maks (px)", 64, 512, 256, step=64)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)


# ---------------- Helpers ----------------
def preprocess(img, max_side, median_size, alpha_thr):
    img = img.convert("RGBA")
    img.thumbnail((max_side, max_side))
    arr = np.array(img)
    mask = arr[..., 3] >= alpha_thr
    if mask.sum() == 0:
        return None, None
    # isi piksel transparan dengan rata-rata warna agar tidak "bocor" saat median filter
    arr[~mask, :3] = arr[mask, :3].mean(axis=0).astype(np.uint8)
    rgb = Image.fromarray(arr[..., :3])
    if median_size > 1:
        rgb = rgb.filter(ImageFilter.MedianFilter(median_size))
    pixels = np.array(rgb)[mask]
    return rgb, pixels


def to_space(pix_rgb, space):
    x = (pix_rgb / 255.0).reshape(-1, 1, 3)
    if space == "LAB":
        return rgb2lab(x).reshape(-1, 3)
    h, s, v = rgb2hsv(x).reshape(-1, 3).T  # hue sirkular -> embed cos/sin
    ang = h * 2 * np.pi
    return np.stack([s * np.cos(ang), s * np.sin(ang), v], axis=1)


def from_space(c, space):
    if space == "LAB":
        rgb = lab2rgb(c.reshape(-1, 1, 3)).reshape(-1, 3)
    else:
        h = (np.arctan2(c[:, 1], c[:, 0]) / (2 * np.pi)) % 1.0
        s = np.clip(np.hypot(c[:, 0], c[:, 1]), 0, 1)
        v = np.clip(c[:, 2], 0, 1)
        rgb = hsv2rgb(np.stack([h, s, v], axis=1).reshape(-1, 1, 3)).reshape(-1, 3)
    return np.clip(rgb * 255, 0, 255).round().astype(int)


def hex_(c):
    return "#{:02X}{:02X}{:02X}".format(*[int(v) for v in c])


def lum(c):
    def f(v):
        v = v / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = [f(v) for v in c]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = sorted([lum(a), lum(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


def sat(c):
    mx, mn = max(c), min(c)
    return 0 if mx == 0 else (mx - mn) / mx


def mix(a, b, t):
    return [int(a[i] * (1 - t) + b[i] * t) for i in range(3)]


def assign_roles(centers, props):
    order = list(np.argsort(-props))
    bg = order[0]
    rest = order[1:]
    # Text: warna palet dengan kontras >= 4.5 thd background, kalau tidak ada -> hitam/putih
    best = max(rest, key=lambda i: contrast(centers[i], centers[bg])) if rest else None
    if best is not None and contrast(centers[best], centers[bg]) >= 4.5:
        text = centers[best]
        rest.remove(best)
    else:
        text = np.array([0, 0, 0]) if contrast([0, 0, 0], centers[bg]) > contrast([255] * 3, centers[bg]) else np.array([255] * 3)
    accent = max(rest, key=lambda i: sat(centers[i])) if rest else bg
    rest2 = [i for i in rest if i != accent]
    primary = rest2[0] if rest2 else accent
    secondary = rest2[1] if len(rest2) > 1 else primary
    return {
        "Background": centers[bg], "Primary": centers[primary],
        "Secondary": centers[secondary], "Accent": centers[accent], "Text": np.array(text),
    }


def swatch(hexv, label, w="100%"):
    rgb = [int(hexv[i:i + 2], 16) for i in (1, 3, 5)]
    fg = "#000" if contrast(rgb, [0, 0, 0]) > contrast(rgb, [255] * 3) else "#fff"
    return (f"<div style='background:{hexv};color:{fg};padding:18px 8px;border-radius:10px;"
            f"text-align:center;width:{w};font-family:monospace'><b>{label}</b><br>{hexv}</div>")


def theme_card(name, bg, text, primary, secondary, accent):
    btn_fg = "#000" if contrast(hex_to_rgb(primary), [0, 0, 0]) > contrast(hex_to_rgb(primary), [255] * 3) else "#fff"
    return f"""
    <div style='background:{bg};color:{text};padding:20px;border-radius:14px;border:1px solid {secondary}'>
      <h3 style='margin:0 0 6px 0;color:{text}'>{name} Mode</h3>
      <p style='margin:0 0 12px 0'>Preview overlay stream / UI card.</p>
      <span style='background:{primary};color:{btn_fg};padding:6px 14px;border-radius:8px;margin-right:8px'>Primary</span>
      <span style='background:{secondary};color:{text};padding:6px 14px;border-radius:8px;margin-right:8px'>Secondary</span>
      <span style='border:2px solid {accent};color:{accent};padding:4px 12px;border-radius:8px'>Accent</span>
    </div>"""


def hex_to_rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


# ---------------- Main ----------------
file = st.file_uploader("Upload avatar VTuber (PNG, disarankan transparan)", type=["png", "webp", "jpg", "jpeg"])

if file:
    original = Image.open(file)
    filtered, pixels = preprocess(original, max_side, median_size, alpha_thr)
    if pixels is None:
        st.error("Semua piksel transparan. Turunkan ambang alpha atau pakai gambar lain.")
        st.stop()

    c1, c2 = st.columns(2)
    c1.image(original, caption="Original", use_container_width=True)
    c2.image(filtered, caption=f"Setelah Median Filter ({median_size}x{median_size})", use_container_width=True)
    st.write(f"Piksel valid (non-transparan): **{len(pixels):,}**")

    with st.spinner("Menjalankan K-Means..."):
        feats = to_space(pixels, space)
        km = KMeans(n_clusters=K, n_init=10, random_state=int(seed)).fit(feats)
        centers = from_space(km.cluster_centers_, space)
        props = np.bincount(km.labels_, minlength=K) / len(km.labels_)

    # Palet mentah
    st.subheader("Palet Dominan (centroid K-Means)")
    order = np.argsort(-props)
    cols = st.columns(K)
    for col, i in zip(cols, order):
        col.markdown(swatch(hex_(centers[i]), f"{props[i]*100:.1f}%"), unsafe_allow_html=True)

    # Role
    roles = assign_roles(centers, props)
    st.subheader("Peran Warna")
    cols = st.columns(5)
    for col, (name, c) in zip(cols, roles.items()):
        col.markdown(swatch(hex_(c), name), unsafe_allow_html=True)

    # Light / Dark theme
    bg, tx = roles["Background"], roles["Text"]
    light_bg, dark_bg = mix(bg, [255] * 3, 0.88), mix(bg, [0] * 3, 0.85)
    light_tx = [20, 20, 20] if contrast([20] * 3, light_bg) >= 4.5 else [0, 0, 0]
    dark_tx = [240] * 3
    st.subheader("Preview Tema")
    t1, t2 = st.columns(2)
    args = (hex_(roles["Primary"]), hex_(roles["Secondary"]), hex_(roles["Accent"]))
    t1.markdown(theme_card("Light", hex_(light_bg), hex_(light_tx), *args), unsafe_allow_html=True)
    t2.markdown(theme_card("Dark", hex_(dark_bg), hex_(dark_tx), *args), unsafe_allow_html=True)
    st.caption(f"Kontras teks/background — Light: {contrast(light_tx, light_bg):.1f}:1, Dark: {contrast(dark_tx, dark_bg):.1f}:1 (WCAG AA ≥ 4.5)")

    # Export
    theme = {
        "palette": {k: hex_(v) for k, v in roles.items()},
        "light": {"background": hex_(light_bg), "text": hex_(light_tx)},
        "dark": {"background": hex_(dark_bg), "text": hex_(dark_tx)},
    }
    css = (":root {\n  --primary: %s;\n  --secondary: %s;\n  --accent: %s;\n  --bg: %s;\n  --text: %s;\n}\n"
           "@media (prefers-color-scheme: dark) {\n  :root { --bg: %s; --text: %s; }\n}" %
           (args[0], args[1], args[2], hex_(light_bg), hex_(light_tx), hex_(dark_bg), hex_(dark_tx)))
    st.subheader("Export")
    st.code(css, language="css")
    d1, d2 = st.columns(2)
    d1.download_button("Download theme.json", json.dumps(theme, indent=2), "theme.json", "application/json")
    d2.download_button("Download theme.css", css, "theme.css", "text/css")
else:
    st.info("Upload gambar avatar untuk mulai.")
