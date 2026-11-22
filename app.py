"""Minimal Last-Mile Verifier web app.   python app.py  ->  http://127.0.0.1:5000

Worker  : uploads payment proof + claims (amount, txn id) -> gets a unique verify link
Merchant: opens link, sees tamper score, enters what THEY received -> cross-check
Output  : timestamped JSON record with SHA-256 of the image and of the record itself.
"""
import hashlib, json, os, re, uuid
from datetime import datetime, timezone
from flask import Flask, request, redirect, url_for, render_template_string, abort, send_from_directory
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
UP, REC = "uploads", "records"
os.makedirs(UP, exist_ok=True); os.makedirs(REC, exist_ok=True)
THRESH = 0.5

try:
    from predict import predict_image
    MODEL_OK = os.path.exists("checkpoints/best.pt")
except Exception:  # torch missing
    MODEL_OK = False


def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def sha(b): return hashlib.sha256(b).hexdigest()
def norm_amount(s): return re.sub(r"[^\d.]", "", s or "").rstrip("0").rstrip(".")
def norm_txn(s): return re.sub(r"\W", "", (s or "")).lower()
def rpath(i): return os.path.join(REC, f"{i}.json")


def save_record(r):
    body = {k: v for k, v in r.items() if k != "record_hash"}
    r["record_hash"] = sha(json.dumps(body, sort_keys=True).encode())
    json.dump(r, open(rpath(r["id"]), "w"), indent=2)


PAGE = """<!doctype html><title>Last-Mile Verifier</title>
<body style="font-family:sans-serif;max-width:640px;margin:30px auto;padding:0 12px">
<h2>Last-Mile Verifier</h2>{{ body|safe }}</body>"""

UPLOAD = """<h3>Worker: upload payment proof</h3>
<form method=post enctype=multipart/form-data>
<p>Screenshot: <input type=file name=file accept="image/*" required></p>
<p>Amount claimed: <input name=amount required></p>
<p>Transaction ID: <input name=txn required></p>
<button>Create verification link</button></form>"""


@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template_string(PAGE, body=UPLOAD)
    f = request.files["file"]; raw = f.read()
    rid = uuid.uuid4().hex[:12]
    path = os.path.join(UP, rid + ".jpg")
    img = Image.open(__import__("io").BytesIO(raw)).convert("RGB")
    open(path, "wb").write(raw)
    score = predict_image(img) if MODEL_OK else None
    rec = dict(id=rid, created_utc=now(), image_sha256=sha(raw), tamper_score=score,
               tamper_verdict=None if score is None else ("SUSPICIOUS" if score > THRESH else "NO TAMPERING DETECTED"),
               worker_claim=dict(amount=request.form["amount"], txn=request.form["txn"]),
               merchant_confirmation=None, cross_check=None)
    save_record(rec)
    return redirect(url_for("verify", rid=rid))


@app.route("/v/<rid>", methods=["GET", "POST"])
def verify(rid):
    if not os.path.exists(rpath(rid)): abort(404)
    rec = json.load(open(rpath(rid)))
    if request.method == "POST" and rec["merchant_confirmation"] is None:
        m = dict(amount=request.form["amount"], txn=request.form["txn"], confirmed_utc=now())
        w = rec["worker_claim"]
        rec["merchant_confirmation"] = m
        rec["cross_check"] = dict(amount_match=norm_amount(w["amount"]) == norm_amount(m["amount"]),
                                  txn_match=norm_txn(w["txn"]) == norm_txn(m["txn"]))
        save_record(rec)
    s = rec["tamper_score"]
    body = f"<img src='/img/{rid}' style='max-width:260px'><pre>{json.dumps(rec, indent=2)}</pre>"
    if s is None: body += "<p><b>Model not loaded</b> - train first (see README).</p>"
    if rec["merchant_confirmation"] is None:
        body += "<h3>Merchant / customer: confirm what you received</h3><form method=post><p>Amount: <input name=amount required></p><p>Txn ID: <input name=txn required></p><button>Confirm</button></form>"
    else:
        c = rec["cross_check"]; ok = c["amount_match"] and c["txn_match"]
        body += f"<h3>Cross-check: {'MATCH' if ok else 'MISMATCH'}</h3>"
    return render_template_string(PAGE, body=body)


@app.route("/img/<rid>")
def img(rid): return send_from_directory(UP, rid + ".jpg")


@app.route("/record/<rid>.json")
def record(rid):
    if not os.path.exists(rpath(rid)): abort(404)
    return send_from_directory(REC, rid + ".json")


if __name__ == "__main__":
    app.run(debug=True)
