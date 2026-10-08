# Interactive Neural Network Learning Demo (TensorFlow.js)

A browser-based demo where you choose a dataset, design a small neural network and watch it learn
in real time: decision boundary, loss curves, accuracy and learned weights.

## Files
| File | Purpose |
|------|---------|
| `index.html` | Page layout, controls, loads TensorFlow.js from CDN |
| `style.css`  | Styling |
| `app.js`     | Data generation, TF.js model, training loop, visualisation |

## Run locally
```bash
cd nn-demo
python -m http.server 8000      # or: npx serve
# open http://localhost:8000
```
(You can also double-click `index.html`; an internet connection is needed for the TF.js CDN.)

## Deploy (free)
**GitHub Pages**
1. Create a GitHub repo and upload the 3 files (+ this README).
2. Repo → Settings → Pages → Source: `main` branch, `/ (root)` → Save.
3. Your demo goes live at `https://<username>.github.io/<repo>/`.

**Netlify (drag & drop)**: go to app.netlify.com/drop and drop the `nn-demo` folder.

## How it works
- **Data:** circle / XOR / Gaussian / spiral points, 80% train, 20% test.
- **Model:** `tf.sequential()` of Dense layers; final Dense(1, sigmoid).
- **Training:** Adam or SGD, binary cross-entropy loss, one `model.fit` epoch per animation frame.
- **Visualisation:** the model predicts on a 50×50 grid to paint the decision boundary.
