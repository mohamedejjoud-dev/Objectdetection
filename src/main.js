import { FilesetResolver, ObjectDetector } from '@mediapipe/tasks-vision';

// Per-category box colors
const COLORS = {
  person:       { hex: '#00d4ff', border: 'rgba(0,212,255,0.9)',   bg: 'rgba(0,212,255,0.08)',   glow: '#00d4ff' },
  'cell phone': { hex: '#f59e0b', border: 'rgba(245,158,11,0.9)',  bg: 'rgba(245,158,11,0.08)',  glow: '#f59e0b' },
  laptop:       { hex: '#a855f7', border: 'rgba(168,85,247,0.9)',  bg: 'rgba(168,85,247,0.08)',  glow: '#a855f7' },
  keyboard:     { hex: '#93c5fd', border: 'rgba(147,197,253,0.9)', bg: 'rgba(147,197,253,0.08)', glow: '#93c5fd' },
  mouse:        { hex: '#c084fc', border: 'rgba(192,132,252,0.9)', bg: 'rgba(192,132,252,0.08)', glow: '#c084fc' },
  remote:       { hex: '#fbbf24', border: 'rgba(251,191,36,0.9)',  bg: 'rgba(251,191,36,0.08)',  glow: '#fbbf24' },
  tv:           { hex: '#818cf8', border: 'rgba(129,140,248,0.9)', bg: 'rgba(129,140,248,0.08)', glow: '#818cf8' },
  bottle:       { hex: '#22c55e', border: 'rgba(34,197,94,0.9)',   bg: 'rgba(34,197,94,0.08)',   glow: '#22c55e' },
  cup:          { hex: '#fb7185', border: 'rgba(251,113,133,0.9)', bg: 'rgba(251,113,133,0.08)', glow: '#fb7185' },
  book:         { hex: '#f472b6', border: 'rgba(244,114,182,0.9)', bg: 'rgba(244,114,182,0.08)', glow: '#f472b6' },
  chair:        { hex: '#60a5fa', border: 'rgba(96,165,250,0.9)',  bg: 'rgba(96,165,250,0.08)',  glow: '#60a5fa' },
  backpack:     { hex: '#34d399', border: 'rgba(52,211,153,0.9)',  bg: 'rgba(52,211,153,0.08)',  glow: '#34d399' },
  watch:        { hex: '#e2e8f0', border: 'rgba(226,232,240,0.9)', bg: 'rgba(226,232,240,0.08)', glow: '#e2e8f0' },
  clock:        { hex: '#cbd5e1', border: 'rgba(203,213,225,0.9)', bg: 'rgba(203,213,225,0.08)', glow: '#cbd5e1' },
  default:      { hex: '#00d4ff', border: 'rgba(0,212,255,0.85)',  bg: 'rgba(0,212,255,0.06)',   glow: '#00d4ff' },
};

const colorFor = (cat) => COLORS[cat?.toLowerCase()] ?? COLORS.default;

// Classes to never show
const BLOCKED = new Set([
  'person', 'man', 'woman', 'boy', 'girl', 'child', 'human',
  'face', 'head', 'hand', 'arm', 'leg', 'body',
]);

// Require higher confidence for food classes — they're rarely in webcam scenes
// and the model commonly misclassifies watches as donut/clock/cell phone
// (COCO has no 'watch' class, so we fix it via size heuristics below)
const MIN_CONF = {
  donut: 0.72, pizza: 0.72, cake: 0.72,
  sandwich: 0.70, 'hot dog': 0.70,
  banana: 0.68, apple: 0.65, orange: 0.65,
  broccoli: 0.68, carrot: 0.68,
};

// ── Wristwatch remap ─────────────────────────────────────────────────────────
// A watch face is small, close to square, and mid-frame sized. Thresholds are
// tight on purpose: an AirPods charging case (44×53mm ≈ AR 0.82) and a pair of
// buds (≈0.2% of frame) fall outside them, so they no longer inherit "watch".
const WATCH_MIN_AREA = 0.0025;   // smaller than a watch face at webcam distance
const WATCH_MAX_AREA = 0.025;    // a watch is never this big in frame
const WATCH_AR_MIN   = 0.85;     // face-only boxes: round (1.0) or square (≈1.0)
const WATCH_AR_MAX   = 1.15;

// ── Cell phone silhouette gate ─────────────────────────────────────────────────
// COCO has no earbuds class, so when the model sees AirPods its best available
// guess is `cell phone`. Geometry separates the two reliably enough:
//   • a real phone is never smaller than PHONE_MIN_AREA in a webcam frame,
//   • a phone held upright or lying flat is clearly rectangular — outside
//     PHONE_AR_MIN/MAX — while a charging case sits at AR ≈ 0.75–1.3,
//   • only a very large squarish box (a rotated phone up close) is allowed
//     through the squarish path.
// Anything failing these checks is dropped rather than captioned wrong.
const PHONE_MIN_AREA    = 0.0075; // <0.75% of frame ⇒ buds / case, not a phone
const PHONE_AR_MIN      = 0.75;   // portrait phone ≈ 0.5, case ≈ 0.82
const PHONE_AR_MAX      = 1.40;   // landscape phone ≈ 2.0, case-on-side ≈ 1.25
const PHONE_SQUARISH_OK = 0.05;   // squarish box ≥5% of frame = rotated phone up close

// Filters + relabels raw detections. Returns a new array (may drop entries).
function fixClasses(detections, frameArea) {
  const out = [];
  for (const d of detections) {
    const cat = (d.categories?.[0]?.categoryName ?? '').toLowerCase();
    const b = d.boundingBox;
    const ar = b.width / (b.height || 1);
    const frac = (b.width * b.height) / frameArea;
    const watchSized =
      frac >= WATCH_MIN_AREA && frac < WATCH_MAX_AREA &&
      ar >= WATCH_AR_MIN && ar <= WATCH_AR_MAX;

    if ((cat === 'donut' || cat === 'clock' || cat === 'cell phone') && watchSized) {
      out.push({ ...d, categories: [{ ...d.categories[0], categoryName: 'watch' }] });
      continue;
    }

    if (cat === 'cell phone') {
      const phoneShaped = ar <= PHONE_AR_MIN || ar >= PHONE_AR_MAX;
      const bigEnough   = frac >= PHONE_MIN_AREA;
      const squarishBig = frac >= PHONE_SQUARISH_OK;
      if (!(bigEnough && (phoneShaped || squarishBig))) continue; // drop, don't mislabel
    }

    out.push(d);
  }
  return out;
}

class CentroidTracker {
  constructor({ maxDisappeared = 5, iouThreshold = 0.30, confirmFrames = 3, boxAlpha = 0.35, scoreAlpha = 0.40, labelStableFrames = 2 } = {}) {
    this.nextId = 0;
    this.objects = new Map();
    this.maxDisappeared = maxDisappeared;
    this.iouThreshold = iouThreshold;
    this.confirmFrames = confirmFrames;
    this.boxAlpha = boxAlpha;
    this.scoreAlpha = scoreAlpha;
    this.labelStableFrames = labelStableFrames;
  }

  static iou(a, b) {
    const ix1 = Math.max(a.originX, b.originX);
    const iy1 = Math.max(a.originY, b.originY);
    const ix2 = Math.min(a.originX + a.width,  b.originX + b.width);
    const iy2 = Math.min(a.originY + a.height, b.originY + b.height);
    if (ix2 <= ix1 || iy2 <= iy1) return 0;
    const inter = (ix2 - ix1) * (iy2 - iy1);
    const union = a.width * a.height + b.width * b.height - inter;
    return union > 0 ? inter / union : 0;
  }

  static smoothBox(stored, raw, alpha) {
    stored.originX = stored.originX * (1 - alpha) + raw.originX * alpha;
    stored.originY = stored.originY * (1 - alpha) + raw.originY * alpha;
    stored.width   = stored.width   * (1 - alpha) + raw.width   * alpha;
    stored.height  = stored.height  * (1 - alpha) + raw.height  * alpha;
  }

  update(detections) {
    if (detections.length === 0) {
      for (const [id, obj] of this.objects) {
        obj.disappeared++;
        if (obj.disappeared > this.maxDisappeared) this.objects.delete(id);
      }
      return this._output();
    }

    if (this.objects.size === 0) {
      for (const det of detections) this._register(det);
      return this._output();
    }

    const ids  = [...this.objects.keys()];
    const objs = ids.map(id => this.objects.get(id));
    const matched = new Set();
    const used    = new Set();

    const pairs = [];
    for (let ei = 0; ei < objs.length; ei++) {
      for (let di = 0; di < detections.length; di++) {
        const score = CentroidTracker.iou(objs[ei].box, detections[di].boundingBox);
        if (score >= this.iouThreshold) pairs.push({ ei, di, score });
      }
    }
    pairs.sort((a, b) => b.score - a.score);

    for (const { ei, di } of pairs) {
      const id = ids[ei];
      if (used.has(id) || matched.has(di)) continue;

      const obj = this.objects.get(id);
      const det = detections[di];
      const rawScore = det.categories?.[0]?.score ?? 0;
      const rawCat   = det.categories?.[0]?.categoryName ?? obj.category;

      CentroidTracker.smoothBox(obj.box, det.boundingBox, this.boxAlpha);
      const prevScore = obj.score;
      obj.score = prevScore * (1 - this.scoreAlpha) + rawScore * this.scoreAlpha;

      // Label hysteresis. A new category must (a) hold for several consecutive
      // frames and (b) actually beat the score we already trust, before we switch.
      // This stops the caption flickering between plausible classes
      // (cup/bottle, remote/cell phone, book/laptop, …).
      if (rawCat === obj.category) {
        obj.pendingCat = null;
        obj.pendingFrames = 0;
      } else if (rawCat === obj.pendingCat) {
        obj.pendingFrames++;
      } else {
        obj.pendingCat = rawCat;
        obj.pendingFrames = 1;
      }

      if (obj.pendingFrames >= this.labelStableFrames && rawScore >= prevScore + 0.05) {
        obj.category = rawCat;
        obj.pendingCat = null;
        obj.pendingFrames = 0;
      }

      obj.cx = obj.box.originX + obj.box.width  / 2;
      obj.cy = obj.box.originY + obj.box.height / 2;
      obj.disappeared = 0;
      obj.seenFrames  = Math.min((obj.seenFrames ?? 0) + 1, 12);

      matched.add(di);
      used.add(id);
    }

    for (const id of ids) {
      if (!used.has(id)) {
        const obj = this.objects.get(id);
        obj.disappeared++;
        if (obj.disappeared > this.maxDisappeared) this.objects.delete(id);
      }
    }

    for (let di = 0; di < detections.length; di++) {
      if (!matched.has(di)) this._register(detections[di]);
    }

    return this._output();
  }

  _register(det) {
    const b = det.boundingBox;
    this.objects.set(this.nextId++, {
      id: this.nextId - 1,
      cx: b.originX + b.width  / 2,
      cy: b.originY + b.height / 2,
      box: { originX: b.originX, originY: b.originY, width: b.width, height: b.height },
      category:    det.categories?.[0]?.categoryName ?? 'object',
      score:       det.categories?.[0]?.score ?? 0,
      pendingCat:  null,
      pendingFrames: 0,
      disappeared: 0,
      seenFrames:  1,
    });
  }

  // Evidence-based confirmation. A confidently-scored object must hold for 3
  // frames; a weak-but-plausible one (e.g. an oddly-shaped box the model only
  // half-recognises) must hold for 5–8 frames instead of being hard-thresholded
  // away. Real objects stay put long enough to accumulate the evidence; noise
  // flickers in and out before it can.
  _requiredFrames(score) {
    if (score >= 0.50) return this.confirmFrames;  // 3 frames — confident
    if (score >= 0.35) return Math.max(this.confirmFrames, 5);
    return 8;                                      // ~0.25s of proof for the weakest
  }

  _output() {
    return [...this.objects.values()].filter(
      o => o.disappeared === 0 && o.seenFrames >= this._requiredFrames(o.score)
    );
  }

  reset() { this.objects.clear(); this.nextId = 0; }
}

class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: true, willReadFrequently: false });
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  draw(obj) {
    const { box, category, score, id } = obj;
    const { originX, originY, width, height } = box;
    if (width <= 2 || height <= 2) return;

    // Canvas is not mirrored but video is (CSS scaleX(-1)), so flip X here
    const mx  = this.canvas.width - originX - width;
    const col = colorFor(category);
    const ctx = this.ctx;

    ctx.save();

    ctx.fillStyle = col.bg;
    ctx.fillRect(mx, originY, width, height);

    ctx.strokeStyle = col.border;
    ctx.lineWidth   = 1.8;
    ctx.shadowColor = col.glow;
    ctx.shadowBlur  = 8;
    ctx.strokeRect(mx, originY, width, height);

    // Corner brackets
    const cl = Math.min(18, width * 0.22, height * 0.22);
    ctx.shadowBlur  = 0;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 3;
    const corners = [
      [mx, originY + cl, mx, originY, mx + cl, originY],
      [mx + width - cl, originY, mx + width, originY, mx + width, originY + cl],
      [mx, originY + height - cl, mx, originY + height, mx + cl, originY + height],
      [mx + width - cl, originY + height, mx + width, originY + height, mx + width, originY + height - cl],
    ];
    for (const [x1, y1, x2, y2, x3, y3] of corners) {
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.lineTo(x3, y3); ctx.stroke();
    }

    // Label badge
    const label  = `${category.toUpperCase()}  ${Math.round(score * 100)}%`;
    ctx.font     = 'bold 12px "JetBrains Mono", monospace';
    ctx.shadowBlur = 0;
    const badgeW = ctx.measureText(label).width + 18;
    const badgeH = 22;
    const badgeX = mx;
    const badgeY = originY > badgeH ? originY - badgeH : originY + height;

    ctx.fillStyle   = 'rgba(8,12,18,0.92)';
    ctx.strokeStyle = col.border;
    ctx.lineWidth   = 1.2;
    ctx.beginPath();
    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#fff';
    ctx.fillText(label, badgeX + 9, badgeY + 15);

    // Tracking ID chip
    if (id !== undefined) {
      const idLabel = `#${id}`;
      ctx.font = '10px "JetBrains Mono", monospace';
      const idW = ctx.measureText(idLabel).width + 10;
      ctx.fillStyle = col.border;
      ctx.beginPath();
      ctx.roundRect(mx + width - idW - 2, originY + 2, idW, 16, 3);
      ctx.fill();
      ctx.fillStyle = '#000';
      ctx.fillText(idLabel, mx + width - idW + 3, originY + 13);
    }

    ctx.restore();
  }

  drawAll(objects) {
    this.clear();
    for (const obj of objects) this.draw(obj);
  }
}

class Detector {
  constructor() {
    this.video   = document.getElementById('webcam-video');
    this.canvas  = document.getElementById('detection-canvas');
    this.overlay = document.getElementById('status-overlay');
    this.msg     = document.getElementById('status-message');

    this.detector  = null;
    this.vision    = null;
    this.stream    = null;
    this.raf       = null;
    this.running   = false;
    this.lastTs    = -1;
    this.frameCount = 0;
    this.lastFpsTs  = performance.now();
    this.lastVideoTime = -1;
    this.inferCount = 0;
    this.inferTimes = [];
    this.modelTuned = false;

    // Model threshold is low (0.15) so we get more candidates; precision comes
    // from the tracker, not from a hard cut-off: `CentroidTracker._output()`
    // demands more consecutive frames the weaker a detection's score is.
    // displayThreshold is our own filter applied after class corrections.
    this.conf = { score: 0.15, display: 0.28, iou: 0.50, maxResults: 15 };

    this.tracker  = new CentroidTracker({ maxDisappeared: 6, iouThreshold: 0.25, confirmFrames: 3, boxAlpha: 0.40, scoreAlpha: 0.45, labelStableFrames: 2 });
    this.renderer = new Renderer(this.canvas);
  }

  async init() {
    this._status('Loading model…');
    try {
      this.vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );
      await this._loadModel();
      await this._startCamera();
    } catch (err) {
      console.error('[Detector] init:', err);
      this._status(err.message ?? 'Failed to load.', true);
    }
  }

  // Lite2 (512px input) detects materially more objects more correctly than
  // Lite0 (320px input), so it is the default. Lite0 remains the fallback for
  // devices where Lite2 is too slow — see _trackInference().
  async _loadModel(kind = 'accurate') {
    if (this.detector) {
      try { this.detector.close(); } catch (_) {}
      this.detector = null;
    }

    const models = kind === 'fast'
      ? ['https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite']
      : [
          'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite',
          'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
        ];

    for (const url of models) {
      for (const delegate of ['GPU', 'CPU']) {
        try {
          this.detector = await ObjectDetector.createFromOptions(this.vision, {
            baseOptions:    { modelAssetPath: url, delegate },
            scoreThreshold: this.conf.score,
            runningMode:    'VIDEO',
            maxResults:     this.conf.maxResults,
          });
          console.log(`[Detector] loaded ${url.split('/').at(-1)} on ${delegate}`);
          return;
        } catch (e) {
          console.warn(`[Detector] ${delegate} failed:`, e.message);
        }
      }
    }
    throw new Error('No model loaded.');
  }

  async _startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      this._status('Camera not supported.');
      return;
    }
    this._status('Starting camera…');
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user',
          focusMode: 'continuous', // keep hunting/blurring from softening the model input
        },
        audio: false,
      });
      this.video.srcObject = this.stream;
      await new Promise((res, rej) => {
        this.video.onloadedmetadata = () => this.video.play().then(res).catch(rej);
        this.video.onerror = rej;
      });
      this._syncCanvas();
      this.running = true;
      this.lastTs  = -1;
      this.lastVideoTime = -1;
      this.tracker.reset();
      this._hideOverlay();
      this._loop();
    } catch (err) {
      const msg = err.name === 'NotAllowedError' ? 'Camera permission denied.'
                : err.name === 'NotFoundError'   ? 'No camera found.'
                : `Camera error: ${err.message}`;
      this._status(msg, true);
    }
  }

  _loop() {
    const tick = () => {
      if (!this.running) return;
      this._tick();
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  _tick() {
    const now = performance.now();

    this.frameCount++;
    if (now - this.lastFpsTs >= 600) {
      this.lastFpsTs  = now;
      this.frameCount = 0;
    }

    if (!this.detector || this.video.readyState < 2 || this.video.paused || this.video.videoWidth === 0) return;

    // Run inference only on genuinely new camera frames. Re-detecting the same
    // frame at display refresh rate wastes the time budget and makes the temporal
    // filters (confirmation, label hysteresis) count duplicates as evidence.
    if (this.video.currentTime === this.lastVideoTime) return;
    this.lastVideoTime = this.video.currentTime;

    this._syncCanvas();

    const ts = Math.max(Math.round(now), this.lastTs + 1);
    this.lastTs = ts;

    const t0 = performance.now();
    let raw = [];
    try {
      raw = this.detector.detectForVideo(this.video, ts).detections ?? [];
    } catch (e) {
      console.warn('[Detector] inference:', e);
    }
    this._trackInference(performance.now() - t0);

    const frameArea = (this.canvas.width || 1280) * (this.canvas.height || 720);

    const dets = this._nms(
      fixClasses(
        raw.filter(d => {
          const cat   = d.categories?.[0]?.categoryName ?? '';
          const score = d.categories?.[0]?.score ?? 0;
          if (BLOCKED.has(cat.toLowerCase())) return false;
          const b  = d.boundingBox;
          if (b.width * b.height < frameArea * 0.0008) return false;
          const ar = b.width / b.height;
          if (ar > 10 || ar < 0.1) return false;
          return score >= (MIN_CONF[cat.toLowerCase()] ?? this.conf.display);
        }),
        frameArea
      ),
      this.conf.iou
    );

    this.renderer.drawAll(this.tracker.update(dets));
  }

  // Adaptive model choice: after warm-up, if Lite2 averages slower than ~150ms
  // per frame, swap to Lite0 so the feed stays live. Runs once.
  _trackInference(ms) {
    if (this.modelTuned) return;
    this.inferCount++;
    if (this.inferCount <= 10) return;          // skip model warm-up frames
    this.inferTimes.push(ms);
    if (this.inferTimes.length < 30) return;

    this.modelTuned = true;
    const mean = this.inferTimes.reduce((a, b) => a + b, 0) / this.inferTimes.length;
    if (mean <= 150) return;

    console.warn(`[Detector] ${mean.toFixed(0)}ms/frame — downgrading to EfficientDet-Lite0`);
    if (this.detector) {
      try { this.detector.close(); } catch (_) {}
      this.detector = null;
    }
    this._loadModel('fast').catch(() =>
      this._loadModel('accurate').catch(e => console.warn('[Detector] reload failed:', e))
    );
  }

  _nms(detections, thresh) {
    if (detections.length <= 1) return detections;
    const byCat = new Map();
    for (const d of detections) {
      const cat = d.categories?.[0]?.categoryName ?? '__';
      if (!byCat.has(cat)) byCat.set(cat, []);
      byCat.get(cat).push(d);
    }
    const out = [];
    for (const group of byCat.values()) {
      if (group.length === 1) { out.push(group[0]); continue; }
      const sorted = [...group].sort((a, b) => (b.categories?.[0]?.score ?? 0) - (a.categories?.[0]?.score ?? 0));
      const drop = new Set();
      for (let i = 0; i < sorted.length; i++) {
        if (drop.has(i)) continue;
        out.push(sorted[i]);
        for (let j = i + 1; j < sorted.length; j++) {
          if (!drop.has(j) && CentroidTracker.iou(sorted[i].boundingBox, sorted[j].boundingBox) > thresh) drop.add(j);
        }
      }
    }
    return out;
  }

  _syncCanvas() {
    const vw = this.video.videoWidth  || 1280;
    const vh = this.video.videoHeight || 720;
    if (this.canvas.width !== vw || this.canvas.height !== vh) {
      this.canvas.width  = vw;
      this.canvas.height = vh;
    }
  }

  _hideOverlay() { this.overlay.classList.add('hidden'); }

  _status(msg, isError = false) {
    this.msg.textContent = msg;
    this.overlay.classList.remove('hidden');
    const spinner = this.overlay.querySelector('.status-spinner');
    if (spinner) spinner.style.display = isError ? 'none' : '';
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const app = new Detector();
  app.init();
  window.__app = app;
  window.__fixClasses = fixClasses; // test hook
});
