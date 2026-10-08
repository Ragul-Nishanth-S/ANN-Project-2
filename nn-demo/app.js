'use strict';
/* Interactive Neural Network Learning Demo (TensorFlow.js) */

const $ = (id) => document.getElementById(id);

/* ---------- Input features the network may use ---------- */
const FEATURES = {
  x1:   { label: 'X₁',     fn: (a, b) => a },
  x2:   { label: 'X₂',     fn: (a, b) => b },
  x1sq: { label: 'X₁²',    fn: (a, b) => a * a },
  x2sq: { label: 'X₂²',    fn: (a, b) => b * b },
  x1x2: { label: 'X₁X₂',   fn: (a, b) => a * b },
  sin1: { label: 'sin(X₁)', fn: (a, b) => Math.sin(3 * a) },
  sin2: { label: 'sin(X₂)', fn: (a, b) => Math.sin(3 * b) },
};

/* ---------- Application state ---------- */
const S = {
  dataset: 'circle',
  noise: 0.1,
  nPoints: 300,
  features: ['x1', 'x2', 'x1sq', 'x2sq'],
  hidden: [6, 4],
  activation: 'tanh',
  optimizer: 'adam',
  lr: 0.03,
  batch: 16,
  running: false,
  epoch: 0,
  lossTrain: [],
  lossTest: [],
  model: null,
  trainPts: [], testPts: [],
  trainX: null, trainY: null, testX: null, testY: null,
  stats: { trl: null, tel: null, tra: null, tea: null },
};

/* ---------- Synthetic datasets (coordinates roughly in [-1, 1]) ---------- */
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function rnd(a, b) { return a + Math.random() * (b - a); }

function genData(kind, n, noise) {
  const pts = [];
  const half = Math.floor(n / 2);
  if (kind === 'circle') {
    for (let i = 0; i < n; i++) {
      const inner = i < half;
      const r = inner ? rnd(0, 0.5) : rnd(0.7, 1.0);
      const t = rnd(0, 2 * Math.PI);
      pts.push({ x: r * Math.cos(t) + randn() * noise * 0.4, y: r * Math.sin(t) + randn() * noise * 0.4, label: inner ? 1 : 0 });
    }
  } else if (kind === 'xor') {
    for (let i = 0; i < n; i++) {
      let a = rnd(-1, 1), b = rnd(-1, 1);
      a += a > 0 ? 0.1 : -0.1; b += b > 0 ? 0.1 : -0.1; // gap near axes
      pts.push({ x: a + randn() * noise * 0.4, y: b + randn() * noise * 0.4, label: a * b > 0 ? 1 : 0 });
    }
  } else if (kind === 'gaussian') {
    for (let i = 0; i < n; i++) {
      const c = i < half ? 1 : 0;
      const m = c ? 0.5 : -0.5;
      pts.push({ x: m + randn() * (0.2 + noise), y: m + randn() * (0.2 + noise), label: c });
    }
  } else if (kind === 'spiral') {
    for (let i = 0; i < n; i++) {
      const c = i % 2;
      const k = Math.floor(i / 2) / (n / 2);
      const r = k * 0.9;
      const t = 1.75 * k * 2 * Math.PI + c * Math.PI;
      pts.push({ x: r * Math.sin(t) + randn() * noise * 0.3, y: r * Math.cos(t) + randn() * noise * 0.3, label: c });
    }
  }
  // shuffle
  for (let i = pts.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pts[i], pts[j]] = [pts[j], pts[i]];
  }
  return pts;
}

function featurize(x, y) {
  return S.features.map((k) => FEATURES[k].fn(x, y));
}

/* ---------- Data -> tensors ---------- */
function disposeData() {
  ['trainX', 'trainY', 'testX', 'testY'].forEach((k) => { if (S[k]) { S[k].dispose(); S[k] = null; } });
}
function buildTensors() {
  disposeData();
  const mk = (pts) => ({
    x: tf.tensor2d(pts.map((p) => featurize(p.x, p.y))),
    y: tf.tensor2d(pts.map((p) => [p.label])),
  });
  const tr = mk(S.trainPts), te = mk(S.testPts);
  S.trainX = tr.x; S.trainY = tr.y; S.testX = te.x; S.testY = te.y;
}
function regenerateData() {
  const pts = genData(S.dataset, S.nPoints, S.noise);
  const cut = Math.floor(pts.length * 0.8);
  S.trainPts = pts.slice(0, cut);
  S.testPts = pts.slice(cut);
  buildTensors();
}

/* ---------- Model ---------- */
function makeOptimizer() {
  return S.optimizer === 'adam' ? tf.train.adam(S.lr) : tf.train.sgd(S.lr);
}
function buildModel() {
  if (S.model) { S.model.dispose(); S.model = null; }
  const nIn = S.features.length;
  const m = tf.sequential();
  let first = true;
  const cfg = (extra) => (first ? Object.assign({ inputShape: [nIn] }, extra) : extra);
  for (const units of S.hidden) {
    m.add(tf.layers.dense(cfg({ units, activation: S.activation, kernelInitializer: 'glorotNormal' })));
    first = false;
  }
  m.add(tf.layers.dense(cfg({ units: 1, activation: 'sigmoid' })));
  m.compile({ optimizer: makeOptimizer(), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  S.model = m;
}

function resetTraining() {
  stopTraining();
  S.epoch = 0;
  S.lossTrain = [];
  S.lossTest = [];
  S.stats = { trl: null, tel: null, tra: null, tea: null };
  buildTensors();
  buildModel();
  drawAll();
}

/* ---------- Training loop ---------- */
async function trainOneEpoch() {
  const h = await S.model.fit(S.trainX, S.trainY, {
    epochs: 1, batchSize: S.batch, shuffle: true, verbose: 0,
  });
  const acc = h.history.acc ? h.history.acc[0] : h.history.accuracy[0];
  const ev = S.model.evaluate(S.testX, S.testY, { verbose: 0 });
  const testLoss = ev[0].dataSync()[0];
  const testAcc = ev[1].dataSync()[0];
  ev.forEach((t) => t.dispose());
  S.epoch++;
  S.stats = { trl: h.history.loss[0], tel: testLoss, tra: acc, tea: testAcc };
  S.lossTrain.push(S.stats.trl);
  S.lossTest.push(S.stats.tel);
}

async function loop() {
  if (!S.running) return;
  await trainOneEpoch();
  if (S.epoch % 2 === 0 || S.epoch < 20) drawAll();
  updateStats();
  if (S.running) requestAnimationFrame(loop);
}
function startTraining() {
  if (S.running) return;
  S.running = true;
  $('btnTrain').textContent = '⏸ Pause';
  loop();
}
function stopTraining() {
  S.running = false;
  $('btnTrain').textContent = '▶ Train';
}

/* ---------- Drawing ---------- */
const COL1 = [245, 158, 11];  // class 1 (orange)
const COL0 = [37, 99, 235];   // class 0 (blue)
const mix = (p) => [0, 1, 2].map((i) => Math.round(COL0[i] * (1 - p) + COL1[i] * p));

function drawBoundary() {
  const cv = $('boundary'), ctx = cv.getContext('2d');
  const W = cv.width, N = 50;
  // predict on a grid
  const grid = [];
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const x = -1.15 + (2.3 * i) / (N - 1);
      const y = 1.15 - (2.3 * j) / (N - 1);
      grid.push(featurize(x, y));
    }
  }
  const preds = tf.tidy(() => S.model.predict(tf.tensor2d(grid)).dataSync());
  const off = document.createElement('canvas');
  off.width = N; off.height = N;
  const octx = off.getContext('2d');
  const img = octx.createImageData(N, N);
  for (let k = 0; k < N * N; k++) {
    const c = mix(preds[k]);
    img.data[k * 4] = c[0]; img.data[k * 4 + 1] = c[1]; img.data[k * 4 + 2] = c[2]; img.data[k * 4 + 3] = 150;
  }
  octx.putImageData(img, 0, 0);
  ctx.clearRect(0, 0, W, W);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, W, W);

  const toPx = (v) => ((v + 1.15) / 2.3) * W;
  const drawPts = (pts, test) => {
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(toPx(p.x), W - toPx(p.y), test ? 4.5 : 3.5, 0, 2 * Math.PI);
      const c = p.label ? COL1 : COL0;
      ctx.fillStyle = test ? '#fff' : `rgb(${c})`;
      ctx.fill();
      ctx.lineWidth = test ? 2 : 1;
      ctx.strokeStyle = test ? `rgb(${c})` : '#fff';
      ctx.stroke();
    }
  };
  drawPts(S.trainPts, false);
  drawPts(S.testPts, true);
}

function drawChart() {
  const cv = $('chart'), ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, pad = 28;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#e5e7eb'; ctx.fillStyle = '#6b7280'; ctx.font = '11px sans-serif';
  const n = S.lossTrain.length;
  const maxL = Math.max(0.8, ...S.lossTrain, ...S.lossTest);
  for (let g = 0; g <= 4; g++) {
    const y = pad + ((H - 2 * pad) * g) / 4;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - 8, y); ctx.stroke();
    ctx.fillText(((maxL * (4 - g)) / 4).toFixed(2), 2, y + 4);
  }
  ctx.fillText('epoch ' + n, W - 70, H - 6);
  const line = (arr, color) => {
    if (arr.length < 2) return;
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
    arr.forEach((v, i) => {
      const x = pad + ((W - pad - 8) * i) / (arr.length - 1);
      const y = H - pad - ((H - 2 * pad) * v) / maxL;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  };
  line(S.lossTrain, '#4f46e5');
  line(S.lossTest, '#ef4444');
}

function drawNetwork() {
  const cv = $('netcanvas'), ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const sizes = [S.features.length, ...S.hidden, 1];
  const xs = sizes.map((_, i) => 30 + ((W - 60) * i) / (sizes.length - 1));
  const ys = (n) => Array.from({ length: n }, (_, k) => (H * (k + 1)) / (n + 1));
  // connections coloured by learned weights
  S.model.layers.forEach((layer, li) => {
    const w = layer.getWeights()[0];
    const data = w.dataSync(); // [in, out]
    const outN = sizes[li + 1];
    const a = ys(sizes[li]), b = ys(outN);
    for (let i = 0; i < sizes[li]; i++) {
      for (let o = 0; o < outN; o++) {
        const v = data[i * outN + o];
        ctx.beginPath();
        ctx.moveTo(xs[li], a[i]); ctx.lineTo(xs[li + 1], b[o]);
        ctx.strokeStyle = v >= 0 ? 'rgba(37,99,235,0.55)' : 'rgba(245,158,11,0.65)';
        ctx.lineWidth = Math.min(4, 0.3 + Math.abs(v) * 1.5);
        ctx.stroke();
      }
    }
  });
  // neurons
  sizes.forEach((n, li) => {
    ys(n).forEach((y) => {
      ctx.beginPath(); ctx.arc(xs[li], y, 7, 0, 2 * Math.PI);
      ctx.fillStyle = '#fff'; ctx.fill(); ctx.strokeStyle = '#374151'; ctx.lineWidth = 1.5; ctx.stroke();
    });
  });
}

function updateStats() {
  const f = (v, d = 3) => (v == null ? '–' : v.toFixed(d));
  $('sEpoch').textContent = S.epoch;
  $('sTrainLoss').textContent = f(S.stats.trl);
  $('sTestLoss').textContent = f(S.stats.tel);
  $('sTrainAcc').textContent = S.stats.tra == null ? '–' : (S.stats.tra * 100).toFixed(1) + '%';
  $('sTestAcc').textContent = S.stats.tea == null ? '–' : (S.stats.tea * 100).toFixed(1) + '%';
}
function drawAll() {
  if (!S.model) return;
  drawBoundary(); drawChart(); drawNetwork(); updateStats();
}

/* ---------- UI wiring ---------- */
function renderFeatureChecks() {
  const box = $('features');
  box.innerHTML = '';
  for (const [key, def] of Object.entries(FEATURES)) {
    const lab = document.createElement('label');
    const on = S.features.includes(key);
    lab.className = on ? 'on' : '';
    lab.innerHTML = `<input type="checkbox" ${on ? 'checked' : ''}/>${def.label}`;
    lab.querySelector('input').addEventListener('change', (e) => {
      const next = e.target.checked ? [...S.features, key] : S.features.filter((k) => k !== key);
      if (next.length === 0) { e.target.checked = true; return; } // need >=1 input
      // keep canonical order
      S.features = Object.keys(FEATURES).filter((k) => next.includes(k));
      renderFeatureChecks();
      resetTraining();
    });
    box.appendChild(lab);
  }
}
function renderLayers() {
  const box = $('layers');
  box.innerHTML = '';
  S.hidden.forEach((units, i) => {
    const row = document.createElement('div');
    row.className = 'layer-row';
    row.innerHTML = `<span>Layer ${i + 1} neurons</span><input type="number" min="1" max="8" value="${units}" />`;
    row.querySelector('input').addEventListener('change', (e) => {
      S.hidden[i] = Math.max(1, Math.min(8, parseInt(e.target.value) || 1));
      e.target.value = S.hidden[i];
      resetTraining();
    });
    box.appendChild(row);
  });
  if (S.hidden.length === 0) box.innerHTML = '<div class="layer-row"><span>No hidden layers (logistic regression)</span></div>';
}

function bind() {
  $('dataset').onchange = (e) => { S.dataset = e.target.value; regenerateData(); resetTraining(); };
  $('noise').oninput = (e) => { S.noise = +e.target.value; $('noiseVal').textContent = S.noise.toFixed(2); };
  $('noise').onchange = () => { regenerateData(); resetTraining(); };
  $('nPoints').oninput = (e) => { S.nPoints = +e.target.value; $('nPointsVal').textContent = S.nPoints; };
  $('nPoints').onchange = () => { regenerateData(); resetTraining(); };
  $('btnRegen').onclick = () => { regenerateData(); resetTraining(); };

  $('btnAddLayer').onclick = () => { if (S.hidden.length < 5) { S.hidden.push(4); renderLayers(); resetTraining(); } };
  $('btnRemoveLayer').onclick = () => { if (S.hidden.length > 0) { S.hidden.pop(); renderLayers(); resetTraining(); } };
  $('activation').onchange = (e) => { S.activation = e.target.value; resetTraining(); };

  $('optimizer').onchange = (e) => { S.optimizer = e.target.value; resetTraining(); };
  $('lr').onchange = (e) => {
    S.lr = +e.target.value;
    // recompile keeps current weights, only swaps the optimizer
    S.model.compile({ optimizer: makeOptimizer(), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  };
  $('batch').onchange = (e) => { S.batch = +e.target.value; };

  $('btnTrain').onclick = () => (S.running ? stopTraining() : startTraining());
  $('btnStep').onclick = async () => { stopTraining(); await trainOneEpoch(); drawAll(); };
  $('btnReset').onclick = () => resetTraining();
}

async function init() {
  await tf.ready();
  renderFeatureChecks();
  renderLayers();
  bind();
  regenerateData();
  resetTraining();
}
init();
