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

## Notes

- Prefers GPU delegate, falls back to CPU automatically.
- `watch` is not a COCO class — the model maps wristwatches to `donut`/`cell phone`/`clock` depending on shape. A size heuristic in `fixClasses()` corrects this at runtime.
- People and faces are filtered out entirely.
