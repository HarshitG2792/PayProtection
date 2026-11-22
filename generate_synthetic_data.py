"""Generate a synthetic dataset of genuine vs. tampered UPI-style payment screenshots.

    python generate_synthetic_data.py --n 3000 --out data

Output layout (this is also the layout you should use for REAL data):
    data/train/genuine  data/train/tampered
    data/val/genuine    data/val/tampered
    data/test/genuine   data/test/tampered
"""
import argparse, io, os, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 360, 640
FONT_CANDIDATES = [
    "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "arial.ttf", "Arial.ttf", "arialbd.ttf",
    "LiberationSans-Regular.ttf", "Roboto-Regular.ttf", "Verdana.ttf", "calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
NAMES = ["Rahul Sharma", "Priya Verma", "Amit Singh", "Neha Gupta", "Suresh Kumar", "Anjali Mehta",
         "Vikas Yadav", "Pooja Jain", "Rohit Chauhan", "Sunita Devi", "Mohd Imran", "Kiran Patel"]
BANKS = ["okicici", "oksbi", "ybl", "paytm", "okaxis", "apl"]
THEMES = [((255, 255, 255), (30, 150, 70)), ((245, 245, 250), (95, 37, 159)),
          ((250, 250, 250), (0, 120, 215)), ((255, 255, 255), (230, 80, 40))]


def get_font(size, rng, bold=False):
    cands = FONT_CANDIDATES[:]
    rng.shuffle(cands)
    for name in cands:
        if bold != ("Bold" in name or "bd" in name) and rng.random() < 0.7:
            continue
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def rand_fields(rng):
    amt = rng.choice([rng.randint(10, 999), rng.randint(1000, 25000)])
    return {
        "amount": f"Rs. {amt:,}.00",
        "to": rng.choice(NAMES),
        "upi": f"{rng.randint(6000000000, 9999999999)}@{rng.choice(BANKS)}",
        "txn": "".join(str(rng.randint(0, 9)) for _ in range(12)),
        "date": f"{rng.randint(1,28):02d} {rng.choice(['Jan','Feb','Mar','Apr','Jun','Aug','Oct','Nov'])} 2026, {rng.randint(1,12):02d}:{rng.randint(0,59):02d} {rng.choice(['am','pm'])}",
    }


def make_receipt(rng):
    """Returns (image, {field: bbox}, {field: bg_colour})."""
    bg, accent = rng.choice(THEMES)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 70], fill=accent)
    d.text((20, 22), "Payment Details", fill="white", font=get_font(20, rng, True))
    cx, cy = W // 2, 150
    d.ellipse([cx - 38, cy - 38, cx + 38, cy + 38], fill=accent)
    d.line([cx - 16, cy, cx - 4, cy + 14, cx + 18, cy - 12], fill="white", width=6)
    d.text((cx - 80, 205), "Payment Successful", fill=(40, 40, 40), font=get_font(20, rng, True))

    f = rand_fields(rng)
    boxes, bgs = {}, {}
    base = rng.choice(["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"])
    def font(sz, bold=False):
        for n in (base, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else base):
            try:
                return ImageFont.truetype(n, sz)
            except OSError:
                pass
        return get_font(sz, rng, bold)

    def put(key, xy, text, fnt, fill):
        d.text(xy, text, fill=fill, font=fnt)
        boxes[key] = d.textbbox(xy, text, font=fnt)
        bgs[key] = bg

    put("amount", (cx - 70, 255), f["amount"], font(34, True), (20, 20, 20))
    y = 330
    for label, key in [("To", "to"), ("UPI ID", "upi"), ("Date", "date"), ("Transaction ID", "txn")]:
        d.text((24, y), label, fill=(130, 130, 130), font=font(13))
        put(key, (24, y + 20), f[key], font(17), (30, 30, 30))
        y += 70
    d.line([20, 320, W - 20, 320], fill=(220, 220, 220), width=1)
    d.rectangle([20, 600, W - 20, 625], outline=accent, width=1)
    return img, boxes, bgs


def tamper(img, boxes, bgs, rng):
    img = img.copy()
    d = ImageDraw.Draw(img)
    key = rng.choice(list(boxes))
    x0, y0, x1, y1 = boxes[key]
    pad = 3
    mode = rng.choice(["retype", "retype", "retype", "copymove", "blurpatch"])
    if mode == "retype":
        # cover old text with a (slightly off) background box and type new text
        col = tuple(min(255, max(0, c + rng.randint(-4, 4))) for c in bgs[key])
        d.rectangle([x0 - pad, y0 - pad, x1 + pad + 20, y1 + pad], fill=col)
        f = rand_fields(rng)
        new = {"amount": f["amount"], "to": f["to"], "upi": f["upi"], "txn": f["txn"], "date": f["date"]}[key]
        size = max(10, int((y1 - y0) * rng.uniform(0.85, 1.15)))
        fnt = get_font(size, rng, key == "amount")
        d.text((x0 + rng.randint(-2, 2), y0 + rng.randint(-2, 2)), new, fill=(25, 25, 25), font=fnt)
    elif mode == "copymove":
        patch = img.crop((x0, y0, x1, y1))
        sy = rng.choice([yy for yy in range(330, 560, 10)])
        img.paste(patch, (x0, sy)) if rng.random() < 0.5 else img.paste(patch.filter(ImageFilter.GaussianBlur(0.8)), (x0, y0))
    else:
        patch = img.crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad)).filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.4)))
        img.paste(patch, (x0 - pad, y0 - pad))
        d = ImageDraw.Draw(img)
        d.text((x0, y0), "9", fill=(25, 25, 25), font=get_font(max(10, y1 - y0), rng))
    return img


def jpeg_roundtrip(img, q):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_pair(rng):
    base, boxes, bgs = make_receipt(rng)
    # BOTH classes get the same kind of random compression history, so the model
    # cannot cheat by only looking at "was it compressed?".
    genuine = jpeg_roundtrip(base, rng.randint(70, 95))
    edited = tamper(base if rng.random() < 0.5 else genuine, boxes, bgs, rng)
    edited = jpeg_roundtrip(edited, rng.randint(70, 95))
    if rng.random() < 0.3:  # sometimes a re-share (WhatsApp-like recompression + resize)
        s = rng.uniform(0.8, 1.0)
        for_name = lambda im: jpeg_roundtrip(im.resize((int(W * s), int(H * s))).resize((W, H)), rng.randint(60, 85))
        genuine, edited = for_name(genuine), for_name(edited)
    return genuine, edited


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="number of genuine/tampered PAIRS")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for split, frac in [("train", 0.8), ("val", 0.1), ("test", 0.1)]:
        for cls in ("genuine", "tampered"):
            os.makedirs(os.path.join(a.out, split, cls), exist_ok=True)
        for i in range(int(a.n * frac)):
            g, t = make_pair(rng)
            g.save(os.path.join(a.out, split, "genuine", f"g_{i:05d}.jpg"), quality=95)
            t.save(os.path.join(a.out, split, "tampered", f"t_{i:05d}.jpg"), quality=95)
    print("Done ->", a.out)


if __name__ == "__main__":
    main()
