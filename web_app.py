"""
web_app.py — Flask Web Dashboard (access via browser on same network)
Usage:  python web_app.py [--camera 0] [--port 5000]
"""

from flask import Flask, Response, render_template_string, jsonify, request
import cv2
import threading
import time
import json
import argparse
from detector import BoltCounter, DEFAULT_EXPECTED

app = Flask(__name__)

# ── Global state ───────────────────────────────────────────────
counter: BoltCounter = None
latest_result = None
latest_frame = None
camera_lock = threading.Lock()
config = {
    "expected": DEFAULT_EXPECTED.copy(),
    "conf": 0.35,
    "iou": 0.3,
    "use_sahi": False,
    "alerts": [],
}


# ── MJPEG streaming ────────────────────────────────────────────
def gen_frames():
    global latest_frame
    while True:
        with camera_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    if latest_result is None:
        return jsonify({})
    d = latest_result.to_dict()
    d.pop("detections", None)
    return jsonify(d)


@app.route("/config", methods=["GET", "POST"])
def cfg():
    global config
    if request.method == "POST":
        data = request.json or {}
        if "expected" in data:
            config["expected"] = {k: int(v) for k, v in data["expected"].items()}
            if counter:
                counter.expected = config["expected"]
        if "conf" in data:
            config["conf"] = float(data["conf"])
            if counter:
                counter.conf_threshold = float(data["conf"])
        return jsonify({"ok": True})
    return jsonify(config)


HTML = """<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bolt Counter Station</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
  header { background: #111; border-bottom: 1px solid #1e90ff33;
           padding: 12px 24px; display: flex; align-items: center; gap: 14px; }
  header h1 { font-size: 1.2rem; color: #1e90ff; letter-spacing: 1px; }
  header .badge { background: #1e90ff22; border: 1px solid #1e90ff44;
                  color: #1e90ff; font-size: .75rem; padding: 3px 10px; border-radius: 20px; }
  .container { display: grid; grid-template-columns: 1fr 320px; gap: 16px;
               padding: 16px; max-width: 1400px; margin: auto; }
  #stream-box { background: #111; border-radius: 10px; overflow: hidden;
                border: 1px solid #222; }
  #stream-box img { width: 100%; display: block; }
  .panel { display: flex; flex-direction: column; gap: 12px; }
  .card { background: #111; border: 1px solid #222; border-radius: 10px; padding: 16px; }
  .card h3 { font-size: .8rem; color: #666; text-transform: uppercase;
             letter-spacing: 1px; margin-bottom: 12px; }
  .count-row { display: flex; justify-content: space-between; align-items: center;
               padding: 8px 0; border-bottom: 1px solid #1a1a1a; }
  .count-row:last-child { border-bottom: none; }
  .count-name { color: #aaa; font-size: .9rem; }
  .count-val { font-size: 1.4rem; font-weight: 700; }
  .ok { color: #00e676; }
  .err { color: #ff1744; }
  .total-big { font-size: 3rem; font-weight: 900; color: #1e90ff;
               text-align: center; padding: 12px 0; }
  .alert-box { background: #1a0000; border: 1px solid #ff174444;
               border-radius: 8px; padding: 10px 14px; margin-top: 4px; }
  .alert-item { color: #ff5252; font-size: .85rem; padding: 2px 0; }
  .status-ok  { color: #00e676; font-size: 1.1rem; font-weight: 700; }
  .status-err { color: #ff1744; font-size: 1.1rem; font-weight: 700; }
  .inf-ms { color: #555; font-size: .75rem; text-align: right; margin-top: 6px; }
  input[type=number] { width: 60px; background: #1a1a1a; border: 1px solid #333;
                       color: #eee; padding: 4px 6px; border-radius: 4px;
                       text-align: right; font-size: .9rem; }
  button { background: #1e90ff22; border: 1px solid #1e90ff66; color: #1e90ff;
           padding: 6px 16px; border-radius: 6px; cursor: pointer; margin-top: 8px;
           width: 100%; font-size: .85rem; }
  button:hover { background: #1e90ff44; }
  @media (max-width: 800px) { .container { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>⚙️  BOLT COUNTER</h1>
  <span class="badge" id="backend-badge">—</span>
  <span class="badge" id="fps-badge">— FPS</span>
</header>
<div class="container">
  <div id="stream-box">
    <img src="/video_feed" alt="Live feed">
  </div>
  <div class="panel">

    <div class="card">
      <h3>Total Objects</h3>
      <div class="total-big" id="total">—</div>
      <div id="status-text" style="text-align:center">—</div>
      <div class="inf-ms" id="inf-ms"></div>
    </div>

    <div class="card">
      <h3>Part Count</h3>
      <div id="counts"></div>
    </div>

    <div class="card" id="alert-card" style="display:none">
      <h3>⚠️ Alerts</h3>
      <div id="alerts"></div>
    </div>

    <div class="card">
      <h3>Expected Config</h3>
      <div id="exp-inputs"></div>
      <button onclick="saveConfig()">Save</button>
    </div>

  </div>
</div>

<script>
let expected = {};
let frameCount = 0;
let lastFpsTime = Date.now();

async function loadConfig() {
  const r = await fetch('/config');
  const c = await r.json();
  expected = c.expected || {};
  const el = document.getElementById('exp-inputs');
  el.innerHTML = Object.entries(expected).map(([k, v]) =>
    `<div class="count-row">
      <span class="count-name">${k}</span>
      <input type="number" id="exp_${k}" value="${v}" min="0">
    </div>`
  ).join('');
}

async function saveConfig() {
  const newExp = {};
  for (const k of Object.keys(expected)) {
    newExp[k] = parseInt(document.getElementById('exp_' + k)?.value || 0);
  }
  await fetch('/config', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({expected: newExp})
  });
  expected = newExp;
}

async function poll() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    if (!d || !d.counts) return;

    document.getElementById('total').textContent = d.total_objects ?? '—';
    document.getElementById('inf-ms').textContent =
      `${d.inference_ms ?? '—'} ms · backend: ${d.backend ?? '—'}`;

    const ok = !d.alerts || d.alerts.length === 0;
    document.getElementById('status-text').innerHTML =
      ok ? '<span class="status-ok">✓ ALL OK</span>'
         : '<span class="status-err">✗ MISMATCH</span>';

    const counts = d.counts || {};
    document.getElementById('counts').innerHTML =
      Object.entries(counts).map(([cls, cnt]) => {
        const exp = expected[cls] || 0;
        const clsOk = exp <= 0 || cnt === exp;
        return `<div class="count-row">
          <span class="count-name">${cls}</span>
          <span class="count-val ${clsOk ? 'ok' : 'err'}">
            ${cnt}${exp > 0 ? '/' + exp : ''}
          </span>
        </div>`;
      }).join('') || '<div style="color:#555;font-size:.85rem">No detections</div>';

    const alertCard = document.getElementById('alert-card');
    const alertsEl = document.getElementById('alerts');
    if (d.alerts && d.alerts.length > 0) {
      alertCard.style.display = 'block';
      alertsEl.innerHTML = d.alerts.map(a =>
        `<div class="alert-item">• ${a}</div>`).join('');
    } else {
      alertCard.style.display = 'none';
    }
  } catch(e) {}
}

loadConfig();
setInterval(poll, 500);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


# ── Camera capture thread ──────────────────────────────────────
def capture_loop(camera_id: int, use_hailo: bool, model: str, hailo_model: str,
                 use_sahi: bool = False, sahi_slice: int = 320):
    global counter, latest_result, latest_frame

    counter = BoltCounter(
        model_path=model,
        hailo_model_path=hailo_model,
        conf_threshold=config["conf"],
        iou_threshold=config["iou"],
        expected_counts=config["expected"],
        use_hailo=use_hailo,
        use_sahi=use_sahi,
        sahi_slice_size=sahi_slice,
    )

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        result = counter.process_frame(frame)
        overlay = counter.draw(frame, result)
        with camera_lock:
            latest_frame = overlay
        latest_result = result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--hailo-model", default="models/best.hef")
    parser.add_argument("--no-hailo", action="store_true")
    parser.add_argument("--sahi", action="store_true",
                        help="Enable SAHI for dense object counting")
    parser.add_argument("--sahi-slice", type=int, default=320)
    args = parser.parse_args()

    t = threading.Thread(
        target=capture_loop,
        args=(args.camera, not args.no_hailo, args.model, args.hailo_model,
              args.sahi, args.sahi_slice),
        daemon=True,
    )
    t.start()

    print(f"\n🌐 Web dashboard: http://0.0.0.0:{args.port}")
    print("   (เปิด browser บน PC/Phone ใน network เดียวกัน)")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
