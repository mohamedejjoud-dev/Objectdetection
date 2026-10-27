# Real-Time Object Detection

Webcam object detection in the browser using MediaPipe EfficientDet-Lite. No backend, no install beyond `npm`.

## Stack

- **MediaPipe Tasks Vision** — EfficientDet-Lite0/Lite2 (COCO, 80 classes)
- **Vite** — dev server + bundler

## Run

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) and allow camera access.

## Accuracy

- **EfficientDet-Lite2 by default** (512px input), falling back to Lite0 (320px) if it cannot load. After ~40 frames the app measures real inference time; if Lite2 averages over 150ms/frame it swaps to Lite0 automatically so the feed stays live.
- **Inference runs only on new camera frames**, not once per display refresh, so the temporal filters see real evidence instead of repeated copies of the same frame.
- **Label hysteresis** in `CentroidTracker`: a new class must hold for 2 consecutive frames *and* beat the current score before the caption switches, which stops `cup`/`bottle`, `remote`/`cell phone`, `book`/`laptop` flicker.
- **Evidence-based confirmation instead of a hard cut-off**: the model threshold is low (0.15) and the display filter is only 0.28, so ambiguous objects still reach the tracker — but `CentroidTracker` demands more proof the weaker the score: 3 frames above 0.50, 5 frames from 0.35–0.50, 8 frames below 0.35. Objects the model only half-recognises (a weirdly-shaped box held up to the camera) appear once they stay put briefly, while one-frame noise never does.
- **Continuous focus** is requested from the camera so focus hunting doesn't soften the model input mid-shot.
- **Overlay alignment**: the canvas uses `object-fit: contain` to mirror the video's letterboxing, so boxes stay locked to their objects at any window size.
- **Stricter watch heuristic**: `fixClasses()` only remaps `donut`/`clock`/`cell phone` to `watch` when the box is between 0.25% and 2.5% of the frame *and* nearly square (aspect 0.85–1.15) — tight enough that an AirPods case (aspect ≈ 0.82) and a pair of buds no longer inherit it.
- **Cell phone silhouette gate**: COCO has no earbuds class, so `cell phone` was where AirPods ended up. A `cell phone` box is now dropped unless it is both phone-sized (≥0.75% of frame) and phone-shaped (aspect ≤0.75 or ≥1.40), or squarish and very large (≥5%, a rotated phone up close). AirPods buds and cases fail this and are suppressed instead of captioned `cell phone`.
- Per-class score thresholds (`MIN_CONF`) raise the bar for food classes that the model tends to over-fire on.
- Prefers GPU delegate, falls back to CPU automatically.
- People and faces are filtered out entirely.

### Known limits

MediaPipe's object detector zoo ships only `efficientdet_lite0`, `efficientdet_lite2` and `ssd_mobilenet_v2`, all trained on COCO's 80 classes — none of which is "earbuds". Lite2 is the most accurate of the three, so model swaps cannot make the app say "AirPods"; the gate above instead declines to caption objects the label set cannot describe. Naming them for real requires a custom `.tflite` trained on an earbud-containing dataset, exported with MediaPipe model metadata.
