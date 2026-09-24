# GreggSpeak interactive frontend demo

This directory is a standalone, static portfolio demonstration adapted from the actual GreggSpeak web interface. `styles.css` is a byte-for-byte copy of `greggspeak_ui/backend/static/styles.css`; the header, dashboard, records, and transcript workspace use the same component classes and layout as the source templates. `demo.css` contains only small browser-only additions. It is **not** the running thesis application. The page intentionally uses the source application's normal appearance without a visible demo banner; this README and the main repository README disclose its scope.

The first record uses a team-cleared GreggSpeak test page, its matching segmentation overlay, and the transcript stored by the original application for that same scan. The stored result contains recognition mistakes, which is why the editable review workflow matters. Its 91.46 confidence is a saved model score for that page, not an accuracy claim. The second record is fictional; “Load Example Result” displays predetermined text after a short delay. No OpenCV or model inference runs in this directory. It has no Python, Flask, SQLite, TensorFlow, Raspberry Pi, camera, credentials, or API dependency and makes no requests to an application backend.

The actual assets were copied from `greggspeak_ui/data/page_artifacts/scan_20260522_002416_200bcd34/`, and the matching transcript and page metrics were extracted from the local SQLite batch `batch-fd2f7bf274bb`. Only the cleared page images and selected text/metrics are included. The database and its TSN metadata, including personal contact details, remain excluded. The other shorthand-style illustration is invented and does not encode its fictional transcript.
The case header and empty TSN fields in the interface are neutral examples, not a copy of the source batch's TSN metadata.

## Run locally

Open `index.html` in a browser. The interface uses relative asset paths and hash navigation, so no build step or server is required. For a local HTTP preview, any static file server can serve this directory; for example, from the repository root run `npx serve demo` if Node.js is already installed. Node is optional.

Edits, TSN fields, simulated processing state, and archive/restore actions are stored in this browser's local storage when available. “Reset Data” returns to the initial sample records. If browser storage is blocked, changes still work until the page is refreshed. Do not enter real case or personal data into this public interface preview.

## Deployment

GitHub Pages publishes this directory at [carlharadji.github.io/GreggSpeak](https://carlharadji.github.io/GreggSpeak/) through `.github/workflows/deploy-demo.yml`. The workflow uploads only `demo/` and runs when demo files or the workflow change on `main`. No build step or environment variables are needed.

The demo's hash routes work without server rewrites. It can also be hosted as a static site elsewhere by publishing only this directory.

The hosting page and repository description should identify this as a frontend-only example. Do not deploy the private research workspace as the site root.

## Files

- `index.html`: source-style page shell and navigation.
- `styles.css`: exact copy of the existing web app stylesheet.
- `demo.css`: small styling additions for browser-only controls and placeholders.
- `mock-data.js`: one cleared historical test result plus a fictional record.
- `app.js`: source-layout rendering, hash navigation, browser-local edits, fixed result loading, and TSN fields.
- `assets/gs-logo.svg`: copy of the public GreggSpeak logo already used by the source web app.
- `assets/project-page-001-scanned.png`: exact captured test page from the selected batch.
- `assets/project-page-001-segmentation.png`: exact saved overlay for that page.
- `assets/synthetic-page.svg`: purpose-created illustrative page with invented strokes; no research sample.
- `assets/synthetic-segmentation.svg`: purpose-created overlay illustration, not actual OpenCV output.
