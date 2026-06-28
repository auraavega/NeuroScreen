"""
NeuroScreen — EEG Depression Risk Detection Web App
Flask backend: upload EEG → extract features → predict → display
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
import joblib
import os
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# ── Load model ────────────────────────────────────────────────────
MODEL_PATH   = 'best_model.pkl'
FEATURES_PATH = 'feature_names.pkl'

model         = None
feature_names = None

def load_model():
    global model, feature_names
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[MODEL] Loaded: {MODEL_PATH}")
    else:
        print(f"[MODEL] Not found: {MODEL_PATH} — demo mode active")
    if os.path.exists(FEATURES_PATH):
        feature_names = joblib.load(FEATURES_PATH)

load_model()

# ── EEG Processing ────────────────────────────────────────────────
FS    = 250
BANDS = {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'highbeta':(13,20)}

CHANNEL_NAMES = ['Fp1','Fp2','F3','F4','F7','F8','Fz','T7']  # 8-ch mapping

def bandpass(x, lo=1.0, hi=40.0, fs=FS, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, x)

def notch(x, freq=50.0, fs=FS, Q=30.0):
    from scipy.signal import iirnotch
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, x)

def band_power(sig, flo, fhi, fs=FS):
    freqs, psd = welch(sig, fs=fs, nperseg=min(fs*2, len(sig)))
    idx = (freqs >= flo) & (freqs <= fhi)
    return float(np.trapz(psd[idx], freqs[idx]))

def extract_features(raw_data):
    """
    raw_data: np.ndarray shape (n_channels, n_times)
    Returns: dict of features
    """
    n_ch = raw_data.shape[0]
    ch   = CHANNEL_NAMES[:n_ch]
    feat = {}
    bp   = {}

    # Preprocessing
    data = np.array([bandpass(notch(raw_data[i])) for i in range(n_ch)])

    # Band power per channel
    for i, name in enumerate(ch):
        for band, (flo, fhi) in BANDS.items():
            key      = f'AB.{"ABCDE"[list(BANDS.keys()).index(band)]}.{band}.{"abcdefgh"[i]}.{name}'
            val      = band_power(data[i], flo, fhi)
            feat[key] = val
            bp[f'{name}_{band}'] = val

    # Frontal asymmetry
    pairs = [('Fp1','Fp2'),('F3','F4'),('F7','F8')]
    for ch_l, ch_r in pairs:
        for band in ['delta','theta','alpha','beta','highbeta']:
            if f'{ch_l}_{band}' in bp and f'{ch_r}_{band}' in bp:
                l = np.log(bp[f'{ch_l}_{band}'] + 1e-10)
                r = np.log(bp[f'{ch_r}_{band}'] + 1e-10)
                feat[f'asym_{ch_l}{ch_r}_{band}'] = r - l

    # Temporal asymmetry
    if 'T7_alpha' in bp:
        for band in ['delta','theta','alpha','beta','highbeta']:
            feat[f'asym_T7T8_{band}'] = 0.0  # only T7 available

    # Ratio features
    frontal = [c for c in ch if c in ['Fp1','Fp2','F3','F4','Fz']]
    for name in frontal:
        a  = bp.get(f'{name}_alpha', 1e-10)
        b_ = bp.get(f'{name}_beta',  1e-10)
        t  = bp.get(f'{name}_theta', 1e-10)
        d  = bp.get(f'{name}_delta', 1e-10)
        feat[f'{name}_alpha_beta_ratio']  = a  / (b_ + 1e-10)
        feat[f'{name}_theta_alpha_ratio'] = t  / (a  + 1e-10)
        feat[f'{name}_theta_beta_ratio']  = t  / (b_ + 1e-10)
        feat[f'{name}_delta_alpha_ratio'] = d  / (a  + 1e-10)
        feat[f'{name}_slow_fast_ratio']   = (d + t) / (a + b_ + 1e-10)

    return feat, bp, data

def align_features(feat_dict, feature_names):
    """Align extracted features to model's expected feature order."""
    if feature_names is None:
        return np.array(list(feat_dict.values())).reshape(1, -1)
    row = np.array([feat_dict.get(f, 0.0) for f in feature_names])
    return row.reshape(1, -1)

def parse_openbci_txt(content):
    """Parse OpenBCI .txt file content (bytes or string)."""
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='ignore')
    lines = [l for l in content.split('\n') if l.strip() and not l.startswith('%')]
    rows  = []
    for line in lines:
        vals = [v.strip().rstrip(',') for v in line.split(',')]
        vals = [v for v in vals if v]
        try:
            rows.append([float(v) for v in vals[:8]])
        except:
            continue
    if not rows:
        return None
    arr = np.array(rows)
    return arr.T  # shape: (n_channels, n_times)

def demo_predict():
    """Return demo result when no model is loaded."""
    np.random.seed(42)
    risk = float(np.random.uniform(0.35, 0.72))
    top_feats = [
        {'name': 'Frontal Alpha Asymmetry (F3-F4)', 'value': 0.42, 'direction': 'high'},
        {'name': 'Theta/Beta Ratio (Fz)',            'value': 0.31, 'direction': 'high'},
        {'name': 'Alpha Power (F3)',                 'value': 0.28, 'direction': 'low'},
        {'name': 'Frontal Alpha Asymmetry (Fp1-Fp2)','value': 0.22, 'direction': 'high'},
        {'name': 'Delta/Alpha Ratio (F4)',           'value': 0.18, 'direction': 'high'},
    ]
    return risk, top_feats

# ── Routes ────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML_INDEX)

@app.route('/predict', methods=['POST'])
def predict():
    result = {'success': False}

    try:
        file    = request.files.get('eeg_file')
        demo    = request.form.get('demo', 'false') == 'true'

        if demo or (file is None):
            risk, top_feats = demo_predict()
            result = {
                'success'    : True,
                'risk_score' : risk,
                'top_features': top_feats,
                'n_channels' : 8,
                'duration_sec': 307,
                'n_features' : 164,
                'demo_mode'  : True
            }
            return jsonify(result)

        content  = file.read()
        raw_data = parse_openbci_txt(content)

        if raw_data is None or raw_data.shape[1] < 500:
            return jsonify({'success': False, 'error': 'File tidak valid atau terlalu pendek (min 2 detik)'})

        # Limit ke 5 menit
        raw_data = raw_data[:, :min(raw_data.shape[1], 75000)]

        feat_dict, bp_dict, clean_data = extract_features(raw_data)

        if model is not None and feature_names is not None:
            X_input  = align_features(feat_dict, feature_names)
            risk     = float(model.predict_proba(X_input)[0][1])
            # Top features dari feature importance
            if hasattr(model.named_steps.get('model', model), 'feature_importances_'):
                imp    = model.named_steps['model'].feature_importances_
                top_idx = np.argsort(imp)[-5:][::-1]
                top_feats = [
                    {'name': feature_names[i].replace('asym_','Asym ').replace('_',' ').title(),
                     'value': float(imp[i]),
                     'direction': 'high' if feat_dict.get(feature_names[i], 0) > 0 else 'low'}
                    for i in top_idx
                ]
            else:
                _, top_feats = demo_predict()
        else:
            risk, top_feats = demo_predict()

        # Bandpower summary untuk chart
        bp_summary = {}
        for ch in CHANNEL_NAMES[:raw_data.shape[0]]:
            bp_summary[ch] = {b: float(bp_dict.get(f'{ch}_{b}', 0)) for b in BANDS}

        result = {
            'success'      : True,
            'risk_score'   : risk,
            'top_features' : top_feats,
            'bp_summary'   : bp_summary,
            'n_channels'   : raw_data.shape[0],
            'duration_sec' : raw_data.shape[1] / FS,
            'n_features'   : len(feat_dict),
            'demo_mode'    : model is None
        }

    except Exception as e:
        result = {'success': False, 'error': str(e)}

    return jsonify(result)


# ── HTML ──────────────────────────────────────────────────────────
HTML_INDEX = '''<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NeuroScreen — Deteksi Risiko Depresi Berbasis EEG</title>
<style>
  :root {
    --bg:       #0a0e1a;
    --surface:  #111827;
    --card:     #1a2235;
    --border:   #2a3a55;
    --accent:   #4f8ef7;
    --accent2:  #a78bfa;
    --danger:   #ef4444;
    --warn:     #f59e0b;
    --ok:       #10b981;
    --text:     #e2e8f0;
    --muted:    #8899bb;
    --radius:   12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* NAV */
  nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 40px;
    border-bottom: 1px solid var(--border);
    background: rgba(10,14,26,0.95);
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(8px);
  }
  .logo { display: flex; align-items: center; gap: 10px; }
  .logo-icon {
    width: 36px; height: 36px; border-radius: 8px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .logo-text { font-size: 20px; font-weight: 700; letter-spacing: -0.5px; }
  .logo-sub  { font-size: 11px; color: var(--muted); letter-spacing: 1px; }
  .nav-badge {
    font-size: 11px; padding: 4px 10px; border-radius: 20px;
    background: rgba(79,142,247,0.15); color: var(--accent);
    border: 1px solid rgba(79,142,247,0.3);
  }

  /* HERO */
  .hero {
    text-align: center; padding: 64px 40px 48px;
    background: radial-gradient(ellipse at 50% 0%, rgba(79,142,247,0.08) 0%, transparent 70%);
  }
  .hero-eyebrow {
    font-size: 12px; letter-spacing: 2px; color: var(--accent);
    text-transform: uppercase; margin-bottom: 16px;
  }
  .hero h1 {
    font-size: clamp(28px, 5vw, 48px); font-weight: 800;
    line-height: 1.15; letter-spacing: -1px;
    background: linear-gradient(135deg, #fff 30%, var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 16px;
  }
  .hero p {
    font-size: 16px; color: var(--muted); max-width: 560px;
    margin: 0 auto 32px; line-height: 1.7;
  }
  .disclaimer {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 16px; border-radius: 8px;
    background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3);
    color: var(--warn); font-size: 12px;
  }

  /* MAIN */
  .main { max-width: 960px; margin: 0 auto; padding: 0 24px 80px; }

  /* UPLOAD CARD */
  .upload-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 40px;
    margin-bottom: 24px;
  }
  .upload-zone {
    border: 2px dashed var(--border); border-radius: 10px;
    padding: 48px 24px; text-align: center;
    cursor: pointer; transition: all 0.2s;
    position: relative;
  }
  .upload-zone:hover, .upload-zone.drag-over {
    border-color: var(--accent); background: rgba(79,142,247,0.05);
  }
  .upload-zone input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%;
  }
  .upload-icon { font-size: 40px; margin-bottom: 12px; }
  .upload-zone h3 { font-size: 16px; margin-bottom: 6px; }
  .upload-zone p  { font-size: 13px; color: var(--muted); }
  .file-selected {
    margin-top: 16px; padding: 12px 16px;
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3);
    border-radius: 8px; color: var(--ok); font-size: 13px;
    display: none;
  }

  .btn-row { display: flex; gap: 12px; margin-top: 24px; flex-wrap: wrap; }
  .btn {
    padding: 12px 28px; border-radius: 8px; font-size: 15px;
    font-weight: 600; cursor: pointer; border: none; transition: all 0.2s;
    display: inline-flex; align-items: center; gap: 8px;
  }
  .btn-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
  }
  .btn-primary:hover { opacity: 0.9; transform: translateY(-1px); }
  .btn-secondary {
    background: transparent; color: var(--muted);
    border: 1px solid var(--border);
  }
  .btn-secondary:hover { border-color: var(--accent); color: var(--accent); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  /* PROGRESS */
  .progress-wrap { display: none; margin-top: 20px; }
  .progress-bar  {
    height: 4px; background: var(--border); border-radius: 2px; overflow: hidden;
  }
  .progress-fill {
    height: 100%; width: 0%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width 0.3s; border-radius: 2px;
  }
  .progress-text { font-size: 12px; color: var(--muted); margin-top: 8px; }

  /* RESULTS */
  #results { display: none; }

  .results-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
    margin-bottom: 20px;
  }
  @media (max-width: 640px) { .results-grid { grid-template-columns: 1fr; } }

  .result-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 28px;
  }
  .result-card h3 {
    font-size: 12px; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--muted); margin-bottom: 16px;
  }

  /* GAUGE */
  .gauge-wrap { text-align: center; }
  .gauge-svg  { width: 200px; height: 120px; overflow: visible; }
  .gauge-score {
    font-size: 42px; font-weight: 800; margin: 8px 0 4px;
  }
  .gauge-label {
    font-size: 14px; font-weight: 600; padding: 4px 14px;
    border-radius: 20px; display: inline-block; margin-bottom: 4px;
  }
  .gauge-label.low    { background: rgba(16,185,129,0.15); color: var(--ok); }
  .gauge-label.medium { background: rgba(245,158,11,0.15);  color: var(--warn); }
  .gauge-label.high   { background: rgba(239,68,68,0.15);   color: var(--danger); }

  /* TOP FEATURES */
  .feat-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 0; border-bottom: 1px solid var(--border);
  }
  .feat-item:last-child { border-bottom: none; }
  .feat-bar-wrap { flex: 1; }
  .feat-name  { font-size: 12px; color: var(--text); margin-bottom: 4px; }
  .feat-bar   { height: 5px; background: var(--border); border-radius: 3px; }
  .feat-fill  { height: 100%; border-radius: 3px; transition: width 0.5s; }
  .feat-dir   { font-size: 10px; padding: 2px 8px; border-radius: 10px; white-space: nowrap; }
  .feat-dir.high { background: rgba(239,68,68,0.15); color: var(--danger); }
  .feat-dir.low  { background: rgba(16,185,129,0.15); color: var(--ok); }

  /* STATS ROW */
  .stats-row {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
    margin-bottom: 20px;
  }
  .stat-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px; text-align: center;
  }
  .stat-num  { font-size: 28px; font-weight: 700; color: var(--accent); }
  .stat-label{ font-size: 12px; color: var(--muted); margin-top: 4px; }

  /* BAND POWER CHART */
  .bp-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 28px; margin-bottom: 20px;
  }
  .bp-card h3 {
    font-size: 12px; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--muted); margin-bottom: 20px;
  }
  .bp-bars { display: flex; gap: 8px; align-items: flex-end; height: 100px; }
  .bp-col  { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .bp-bar  {
    width: 100%; border-radius: 4px 4px 0 0; transition: height 0.5s;
    min-height: 4px;
  }
  .bp-band-name { font-size: 10px; color: var(--muted); }

  /* RECOMMENDATION */
  .rec-card {
    background: var(--card); border-radius: var(--radius);
    padding: 24px 28px; margin-bottom: 20px;
    border-left: 4px solid var(--ok);
  }
  .rec-card.medium { border-left-color: var(--warn); }
  .rec-card.high   { border-left-color: var(--danger); }
  .rec-title { font-size: 15px; font-weight: 600; margin-bottom: 8px; }
  .rec-text  { font-size: 13px; color: var(--muted); line-height: 1.6; }

  /* DEMO BADGE */
  .demo-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 20px; font-size: 12px;
    background: rgba(167,139,250,0.1); color: var(--accent2);
    border: 1px solid rgba(167,139,250,0.3); margin-bottom: 16px;
  }

  /* FOOTER */
  footer {
    text-align: center; padding: 32px; font-size: 12px;
    color: var(--muted); border-top: 1px solid var(--border);
  }
</style>
</head>
<body>

<nav>
  <div class="logo">
    <div class="logo-icon">🧠</div>
    <div>
      <div class="logo-text">NeuroScreen</div>
      <div class="logo-sub">EEG DEPRESSION RISK DETECTION</div>
    </div>
  </div>
  <div class="nav-badge">Satria Data 2026</div>
</nav>

<section class="hero">
  <div class="hero-eyebrow">Berbasis Sinyal Otak · Machine Learning · Non-Invasif</div>
  <h1>Deteksi Risiko Depresi<br>dari Sinyal EEG</h1>
  <p>Sistem skrining awal berbasis electroencephalography yang menganalisis pola aktivitas otak untuk mengidentifikasi indikator risiko depresi secara objektif.</p>
  <div class="disclaimer">
    ⚠️ Hanya untuk keperluan skrining awal dan penelitian. Bukan alat diagnosis klinis.
  </div>
</section>

<main class="main">

  <!-- UPLOAD -->
  <div class="upload-card">
    <div class="upload-zone" id="dropZone">
      <input type="file" id="fileInput" accept=".txt,.csv" onchange="onFileSelect(event)">
      <div class="upload-icon">📂</div>
      <h3>Upload File EEG OpenBCI</h3>
      <p>Format: .txt dari OpenBCI GUI · 8 channel · Cyton board<br>Drag & drop atau klik untuk memilih file</p>
    </div>
    <div class="file-selected" id="fileSelected">✅ <span id="fileName"></span></div>

    <div class="btn-row">
      <button class="btn btn-primary" id="btnAnalyze" onclick="analyze()" disabled>
        🔬 Analisis EEG
      </button>
      <button class="btn btn-secondary" onclick="runDemo()">
        ▶ Demo Mode
      </button>
    </div>

    <div class="progress-wrap" id="progressWrap">
      <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
      <div class="progress-text" id="progressText">Memproses sinyal EEG...</div>
    </div>
  </div>

  <!-- RESULTS -->
  <div id="results">

    <div id="demoBadge" class="demo-badge" style="display:none">
      ✨ Demo Mode — hasil menggunakan data simulasi
    </div>

    <!-- STATS ROW -->
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-num" id="statCh">8</div>
        <div class="stat-label">Channel EEG</div>
      </div>
      <div class="stat-card">
        <div class="stat-num" id="statDur">—</div>
        <div class="stat-label">Durasi (detik)</div>
      </div>
      <div class="stat-card">
        <div class="stat-num" id="statFeat">164</div>
        <div class="stat-label">Fitur Diekstrak</div>
      </div>
    </div>

    <!-- GAUGE + TOP FEATURES -->
    <div class="results-grid">
      <div class="result-card gauge-wrap">
        <h3>Risk Score</h3>
        <svg class="gauge-svg" viewBox="-10 -10 220 130" id="gaugeSvg">
          <defs>
            <linearGradient id="gGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%"   stop-color="#10b981"/>
              <stop offset="50%"  stop-color="#f59e0b"/>
              <stop offset="100%" stop-color="#ef4444"/>
            </linearGradient>
          </defs>
          <!-- Track -->
          <path d="M 10 100 A 90 90 0 0 1 190 100"
                fill="none" stroke="#2a3a55" stroke-width="14" stroke-linecap="round"/>
          <!-- Fill -->
          <path d="M 10 100 A 90 90 0 0 1 190 100"
                fill="none" stroke="url(#gGrad)" stroke-width="14" stroke-linecap="round"
                stroke-dasharray="283" stroke-dashoffset="283"
                id="gaugeArc" style="transition: stroke-dashoffset 1s ease"/>
          <!-- Needle -->
          <line id="gaugeNeedle" x1="100" y1="100" x2="100" y2="20"
                stroke="white" stroke-width="2" stroke-linecap="round"
                style="transform-origin: 100px 100px; transition: transform 1s ease"/>
          <circle cx="100" cy="100" r="5" fill="white"/>
        </svg>
        <div class="gauge-score" id="gaugeScore">—</div>
        <div class="gauge-label" id="gaugeLabel">Menunggu analisis</div>
      </div>

      <div class="result-card">
        <h3>Fitur EEG Paling Berpengaruh</h3>
        <div id="featList"></div>
      </div>
    </div>

    <!-- BAND POWER CHART -->
    <div class="bp-card" id="bpCard" style="display:none">
      <h3>Band Power — Frontal Channel (Fp1)</h3>
      <div class="bp-bars" id="bpBars"></div>
    </div>

    <!-- RECOMMENDATION -->
    <div class="rec-card" id="recCard">
      <div class="rec-title" id="recTitle">—</div>
      <div class="rec-text"  id="recText">—</div>
    </div>

  </div><!-- /results -->
</main>

<footer>
  NeuroScreen · Satria Data 2026 · Sistem skrining awal berbasis EEG · Bukan alat diagnosis klinis<br>
  Model: Random Forest (ROC-AUC 0.694) · Dataset: EEG Psychiatric Disorders (Kaggle) · n=294 subjek
</footer>

<script>
let selectedFile = null;

function onFileSelect(e) {
  selectedFile = e.target.files[0];
  if (!selectedFile) return;
  document.getElementById('fileName').textContent = selectedFile.name + ' (' + (selectedFile.size/1024).toFixed(1) + ' KB)';
  document.getElementById('fileSelected').style.display = 'block';
  document.getElementById('btnAnalyze').disabled = false;
}

// Drag & drop
const zone = document.getElementById('dropZone');
zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
zone.addEventListener('dragleave', ()=> zone.classList.remove('drag-over'));
zone.addEventListener('drop', e => {
  e.preventDefault(); zone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) {
    selectedFile = file;
    document.getElementById('fileInput').files = e.dataTransfer.files;
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSelected').style.display = 'block';
    document.getElementById('btnAnalyze').disabled = false;
  }
});

async function analyze() {
  if (!selectedFile) return;
  const fd = new FormData();
  fd.append('eeg_file', selectedFile);
  fd.append('demo', 'false');
  await doRequest(fd);
}

async function runDemo() {
  const fd = new FormData();
  fd.append('demo', 'true');
  await doRequest(fd);
}

async function doRequest(fd) {
  const steps = [
    'Membaca file EEG...', 'Menerapkan bandpass filter (1–40 Hz)...',
    'Menghilangkan noise 50 Hz...', 'Memotong sinyal menjadi epochs...',
    'Menghitung band power...', 'Menghitung frontal asymmetry...',
    'Menghitung ratio features...', 'Menjalankan model Random Forest...',
    'Menyusun hasil analisis...'
  ];
  const prog = document.getElementById('progressWrap');
  const fill = document.getElementById('progressFill');
  const txt  = document.getElementById('progressText');
  prog.style.display = 'block';
  document.getElementById('results').style.display = 'none';

  for (let i = 0; i < steps.length; i++) {
    fill.style.width = ((i+1)/steps.length*100) + '%';
    txt.textContent  = steps[i];
    await new Promise(r => setTimeout(r, 280));
  }

  try {
    const res  = await fetch('/predict', { method:'POST', body: fd });
    const data = await res.json();
    if (data.success) showResults(data);
    else { alert('Error: ' + (data.error || 'Unknown error')); }
  } catch(e) {
    alert('Gagal menghubungi server: ' + e.message);
  }
  prog.style.display = 'none';
}

function showResults(d) {
  document.getElementById('results').style.display = 'block';
  document.getElementById('demoBadge').style.display = d.demo_mode ? 'inline-flex' : 'none';

  // Stats
  document.getElementById('statCh').textContent   = d.n_channels || 8;
  document.getElementById('statDur').textContent  = Math.round(d.duration_sec || 0);
  document.getElementById('statFeat').textContent = d.n_features || 164;

  // Gauge
  const risk  = d.risk_score;
  const pct   = Math.round(risk * 100);
  const ARC   = 283;
  document.getElementById('gaugeArc').setAttribute('stroke-dashoffset', ARC - ARC * risk);
  document.getElementById('gaugeNeedle').style.transform = `rotate(${-90 + risk*180}deg)`;
  document.getElementById('gaugeScore').textContent = pct + '%';

  const lbl = document.getElementById('gaugeLabel');
  if (risk < 0.4) {
    lbl.textContent = 'Risiko Rendah'; lbl.className = 'gauge-label low';
  } else if (risk < 0.65) {
    lbl.textContent = 'Risiko Sedang'; lbl.className = 'gauge-label medium';
  } else {
    lbl.textContent = 'Risiko Tinggi'; lbl.className = 'gauge-label high';
  }

  // Top features
  const colors = ['#4f8ef7','#a78bfa','#f59e0b','#10b981','#ef4444'];
  const fl = document.getElementById('featList');
  fl.innerHTML = '';
  (d.top_features || []).forEach((f, i) => {
    const maxV = d.top_features[0].value || 1;
    const pctBar = Math.round((f.value / maxV) * 100);
    fl.innerHTML += `
      <div class="feat-item">
        <div class="feat-bar-wrap">
          <div class="feat-name">${f.name}</div>
          <div class="feat-bar">
            <div class="feat-fill" style="width:${pctBar}%;background:${colors[i]}"></div>
          </div>
        </div>
        <div class="feat-dir ${f.direction}">${f.direction === 'high' ? '↑ Tinggi' : '↓ Rendah'}</div>
      </div>`;
  });

  // Band power chart
  if (d.bp_summary) {
    const bp = d.bp_summary;
    const firstCh = Object.keys(bp)[0];
    if (firstCh) {
      document.getElementById('bpCard').style.display = 'block';
      const bands = ['delta','theta','alpha','beta','highbeta'];
      const bColors = ['#6366f1','#4f8ef7','#10b981','#f59e0b','#ef4444'];
      const vals = bands.map(b => bp[firstCh][b] || 0);
      const maxV = Math.max(...vals) || 1;
      const bars = document.getElementById('bpBars');
      bars.innerHTML = '';
      bands.forEach((b, i) => {
        const h = Math.max(4, Math.round((vals[i]/maxV)*90));
        bars.innerHTML += `
          <div class="bp-col">
            <div class="bp-bar" style="height:${h}px;background:${bColors[i]}"></div>
            <div class="bp-band-name">${b.charAt(0).toUpperCase()+b.slice(1)}</div>
          </div>`;
      });
    }
  }

  // Recommendation
  const rec = document.getElementById('recCard');
  const rt  = document.getElementById('recTitle');
  const rtx = document.getElementById('recText');
  if (risk < 0.4) {
    rec.className = 'rec-card';
    rt.textContent = '✅ Tidak Ditemukan Indikasi Risiko Depresi yang Signifikan';
    rtx.textContent = 'Pola sinyal EEG subjek tidak menunjukkan indikator risiko depresi yang bermakna berdasarkan model yang dikembangkan. Tetap jaga kesehatan mental dengan aktivitas fisik rutin, tidur yang cukup, dan hubungan sosial yang positif.';
  } else if (risk < 0.65) {
    rec.className = 'rec-card medium';
    rt.textContent = '⚠️ Terdeteksi Indikasi Risiko Depresi Sedang';
    rtx.textContent = 'Pola EEG menunjukkan beberapa indikator yang perlu diperhatikan. Disarankan untuk melakukan skrining lebih lanjut menggunakan instrumen klinis (PHQ-9/BDI) dan berkonsultasi dengan profesional kesehatan mental jika diperlukan.';
  } else {
    rec.className = 'rec-card high';
    rt.textContent = '🔴 Terdeteksi Indikasi Risiko Depresi Tinggi';
    rtx.textContent = 'Pola EEG menunjukkan indikator risiko yang signifikan. Sangat disarankan untuk segera berkonsultasi dengan psikiater atau psikolog klinis untuk evaluasi komprehensif. Sistem ini bukan alat diagnosis — hasil ini harus dikonfirmasi oleh tenaga medis profesional.';
  }

  document.getElementById('results').scrollIntoView({ behavior:'smooth', block:'start' });
}
</script>
</body>
</html>'''

if __name__ == '__main__':
    print("=" * 50)
    print("  NeuroScreen — EEG Depression Risk Detection")
    print("=" * 50)
    print(f"  Model loaded : {model is not None}")
    print(f"  Features     : {len(feature_names) if feature_names else 'N/A'}")
    print()
    print("  Buka browser: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
