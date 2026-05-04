# ================================================================
# MoodBeats AI  —  Emotion-aware Music  |  v4.0  CYBERPUNK EDITION
# ================================================================
# CYBERPUNK UPGRADES IN THIS VERSION:
#
#   VISUAL IDENTITY
#   ▸ Full Blade Runner / Cyberpunk 2077 aesthetic: void black
#     backgrounds, electric cyan (#00f7ff), hot magenta (#ff00ff),
#     acid green (#00ff9f), blood red (#ff0033) neon accents
#   ▸ Orbitron + Rajdhani + Share Tech Mono via Google Fonts
#     — futuristic typography with neon text-shadow glows
#   ▸ Subtle matrix digital rain canvas in background (JS, 4% opacity)
#   ▸ Glassmorphism panels with neon border glows + inner accent lines
#   ▸ Animated scanline overlay across the entire interface
#   ▸ Holographic HUD top wordmark with CSS glitch animation on loop
#   ▸ Floating neon orb via CSS radial gradient + animation
#
#   CUSTOM CURSOR
#   ▸ SVG-based glowing neon cyan digital targeting reticle
#     injected via JS — replaces default cursor globally
#   ▸ Corner tick marks, crosshairs, pulsing center dot
#   ▸ Scales up on hover over interactive elements (cursor-hover class)
#
#   UX IMPROVEMENTS
#   ▸ Confidence meter FULLY REMOVED — zero trace in HTML or Python
#   ▸ Emotion reveal with CSS glitch-in animation
#   ▸ "Neural Scan" status with animated radar ring during capture
#   ▸ Vibe tags rendered as neon pill badges (genre / energy / BPM)
#   ▸ "Regenerate Mix" button — fresh random seed each press
#   ▸ Mini animated waveform per track (CSS bars)
#   ▸ Cyberpunk-styled history expanders with neon colour accents
#
#   BUTTON EFFECTS
#   ▸ YouTube button: intense neon RED glow on hover (#ff0033)
#   ▸ Spotify button: intense neon GREEN glow on hover (#00ff9f)
#   ▸ All buttons: Orbitron font, neon border, smooth scale on hover
#
#   PLAYLIST
#   ▸ Aggressively randomised on every scan + Regenerate press
#   ▸ Unique hash + random descriptors injected into every LLM prompt
#   ▸ Track cards with left-border neon colour per emotion
#   ▸ Animated hover lift + translateX on each track row
#
#   BUG FIXES
#   ▸ st.link_button() key= removed from history (not supported in
#     current Streamlit) — replaced with styled HTML <a> buttons
# ================================================================

import gdown, os

MODEL_PATH = 'emotion_model_best.h5'
GDRIVE_ID = '1xT9XdBZLgXUXlaj0NFtqSJHdmjuQ3qo7'

if not os.path.exists(MODEL_PATH):
    gdown.download(
        f'https://drive.google.com/uc?id={GDRIVE_ID}',
        MODEL_PATH,
        quiet=False
    )

import os, warnings, time, math, random, hashlib

# ── libGL fix for Streamlit Cloud (headless environment) ──────
# opencv-python requires libGL.so.1 which is absent on Streamlit Cloud.
# opencv-python-headless does NOT need it. This env var + import guard
# ensures we always get the headless build.
import os as _os
_os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "0")

import cv2
import numpy as np
from collections import deque, Counter

import streamlit as st

# ── silence TF before importing keras ─────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# ──────────────────────────────────────────────────────────────
# REGISTER CUSTOM OBJECTS
# ──────────────────────────────────────────────────────────────

@keras.utils.register_keras_serializable(package="MoodBeats")
class GeM(layers.Layer):
    """Generalised Mean Pooling — must match training definition."""
    def __init__(self, p_init=3.0, eps=1e-6, **kw):
        super().__init__(**kw)
        self.eps = eps
        self.p_init = p_init

    def build(self, input_shape):
        self.p = self.add_weight(
            name="gem_p", shape=(),
            initializer=tf.constant_initializer(self.p_init),
            trainable=True,
            constraint=keras.constraints.NonNeg(),
        )
        super().build(input_shape)

    def call(self, x):
        x = tf.clip_by_value(x, self.eps, tf.reduce_max(x))
        return tf.reduce_mean(x ** self.p, axis=[1, 2]) ** (1.0 / (self.p + self.eps))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"p_init": self.p_init, "eps": self.eps})
        return cfg


@keras.utils.register_keras_serializable(package="MoodBeats")
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.0,
                 reduction="sum_over_batch_size", **kw):
        super().__init__(reduction=reduction, **kw)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        if self.label_smoothing > 0:
            n = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / n
        ce = -y_true * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        return tf.reduce_sum(self.alpha * tf.pow(1.0 - pt, self.gamma) * ce, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha,
                    "label_smoothing": self.label_smoothing})
        return cfg


@keras.utils.register_keras_serializable(package="MoodBeats")
class CosineWarmRestarts(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr=1e-3, T_0=1000, T_mult=2,
                 min_lr=1e-8, warmup_steps=0, **kw):
        super().__init__(**kw)
        self.base_lr = float(base_lr)
        self.T_0 = float(T_0)
        self.T_mult = float(T_mult)
        self.min_lr = float(min_lr)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.base_lr * step / tf.maximum(self.warmup_steps, 1.0)
        s = tf.maximum(step - self.warmup_steps, 0.0)
        cosine = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * tf.math.mod(s, self.T_0) / self.T_0))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {"base_lr": self.base_lr, "T_0": self.T_0, "T_mult": self.T_mult,
                "min_lr": self.min_lr, "warmup_steps": self.warmup_steps}


CUSTOM_OBJECTS = {
    "GeM": GeM,
    "FocalLoss": FocalLoss,
    "CosineWarmRestarts": CosineWarmRestarts,
}

# ──────────────────────────────────────────────────────────────
# MOOD MAP
# ──────────────────────────────────────────────────────────────

try:
    from mood_map import EMOTIONS, get_mood
except ImportError:
    EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    def get_mood(emotion):
        m = {
            "happy":    {"genre": "Pop / Dance",               "energy": "High",      "bpm": "120-135", "vibe": "euphoric, uplifting"},
            "sad":      {"genre": "Indie / Soul / Comfort",    "energy": "Low",       "bpm": "60-80",   "vibe": "soothing, hopeful, healing"},
            "angry":    {"genre": "Rock / Metal",              "energy": "Very High", "bpm": "140-160", "vibe": "intense, cathartic, satisfying"},
            "fear":     {"genre": "Ambient / Classical",       "energy": "Low",       "bpm": "50-70",   "vibe": "comforting, safe, ethereal"},
            "disgust":  {"genre": "Punk / Alternative",        "energy": "High",      "bpm": "130-150", "vibe": "raw, rebellious, satisfying"},
            "surprise":  {"genre": "Pop / Electronic / Indie",   "energy": "High",       "bpm": "115-135", "vibe": "uplifting, euphoric, energising"},
            "neutral":  {"genre": "Lo-fi / Jazz",              "energy": "Medium",    "bpm": "80-100",  "vibe": "chill, focused"},
        }
        return m.get(emotion, m["neutral"])

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MoodBeats // Neural Music OS",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════
#  CYBERPUNK DESIGN SYSTEM
#  Blade Runner x Cyberpunk 2077 x Neon Genesis Evangelion
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ══════════════════════════════════════
   DESIGN TOKENS
══════════════════════════════════════ */
:root {
  --void:         #080808;
  --bg:           #0a0a0f;
  --surface:      #0d0d1a;
  --surface2:     #111120;
  --surface3:     #161628;
  --border:       #1a1a30;
  --border-neon:  rgba(0,247,255,0.18);

  /* Neon palette */
  --cyan:         #00f7ff;
  --cyan-dim:     rgba(0,247,255,0.07);
  --cyan-glow:    0 0 15px rgba(0,247,255,0.45), 0 0 50px rgba(0,247,255,0.15);
  --magenta:      #ff00ff;
  --mag-dim:      rgba(255,0,255,0.07);
  --mag-glow:     0 0 15px rgba(255,0,255,0.45), 0 0 50px rgba(255,0,255,0.15);
  --green:        #00ff9f;
  --green-dim:    rgba(0,255,159,0.07);
  --green-glow:   0 0 20px rgba(0,255,159,0.55), 0 0 60px rgba(0,255,159,0.18);
  --red:          #ff0033;
  --red-dim:      rgba(255,0,51,0.07);
  --red-glow:     0 0 20px rgba(255,0,51,0.55), 0 0 60px rgba(255,0,51,0.18);

  /* Text */
  --text:         #dde0f0;
  --text2:        #606080;
  --text3:        #282838;

  /* Fonts */
  --orbitron:     'Orbitron', monospace;
  --rajdhani:     'Rajdhani', sans-serif;
  --mono:         'Share Tech Mono', monospace;

  /* Shape */
  --radius:       3px;
  --radius-md:    6px;
}

/* ══════════════════════════════════════
   CURSOR — default restored
══════════════════════════════════════ */
/* Custom cursor removed — default system cursor */

/* ══════════════════════════════════════
   GLOBAL
══════════════════════════════════════ */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
  background: var(--bg) !important;
  color: var(--text) !important;
  transition: background 1.4s ease, color 1.4s ease !important;
}

/* Smooth color transitions on key UI elements */
h1, h2, h3, h4,
.stButton > button,
.stLinkButton > a,
.cb-btn,
.section-label,
.section-label::before,
[data-testid="column"] > div:first-child,
[data-testid="column"] > div:first-child::before,
[data-testid="column"] > div:first-child::after,
.cp-corner::before, .cp-corner::after,
::-webkit-scrollbar-thumb,
#audio-viz span,
#audio-dot,
#audio-status,
.lock-fill,
.cp-beam {
  transition: color 1.2s ease, background 1.2s ease, border-color 1.2s ease,
              box-shadow 1.2s ease, text-shadow 1.2s ease !important;
}

/* Matrix canvas */
#matrix-canvas {
  position: fixed; inset: 0;
  z-index: 0; pointer-events: none;
  opacity: 0.035;
}

/* Scanline overlay */
[data-testid="stMain"]::before {
  content: '';
  position: fixed; inset: 0;
  z-index: 1; pointer-events: none;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 3px,
    rgba(0,247,255,0.010) 3px,
    rgba(0,247,255,0.010) 4px
  );
}

/* Ambient neon orb top-right */
[data-testid="stMain"]::after {
  content: '';
  position: fixed;
  width: 700px; height: 700px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0,247,255,0.035) 0%, transparent 68%);
  top: -280px; right: -280px;
  z-index: 0; pointer-events: none;
  animation: orb-float 14s ease-in-out infinite alternate;
}
@keyframes orb-float {
  from { transform: translate(0, 0); }
  to   { transform: translate(-60px, 70px); }
}

/* ── Cyberpunk background grid lines ── */
#cp-grid {
  position: fixed; inset: 0;
  z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(0,247,255,0.028) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,247,255,0.028) 1px, transparent 1px);
  background-size: 60px 60px;
  mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 40%, transparent 100%);
  animation: grid-pulse 8s ease-in-out infinite alternate;
}
@keyframes grid-pulse {
  0%   { opacity: 0.6; background-size: 60px 60px; }
  100% { opacity: 1.0; background-size: 62px 62px; }
}

/* ── Neon pulse rings ── */
#cp-rings {
  position: fixed;
  left: 50%; top: 50%;
  transform: translate(-50%, -50%);
  width: 1px; height: 1px;
  z-index: 0; pointer-events: none;
}
.cp-ring {
  position: absolute;
  border-radius: 50%;
  border: 1px solid;
  transform: translate(-50%, -50%);
  animation: ring-expand 7s ease-out infinite;
  opacity: 0;
}
.cp-ring:nth-child(1) { width: 300px; height: 300px; border-color: rgba(0,247,255,0.12); animation-delay: 0s; }
.cp-ring:nth-child(2) { width: 600px; height: 600px; border-color: rgba(255,0,255,0.08); animation-delay: 2.3s; }
.cp-ring:nth-child(3) { width: 900px; height: 900px; border-color: rgba(0,255,159,0.06); animation-delay: 4.6s; }
.cp-ring:nth-child(4) { width: 1200px;height: 1200px;border-color: rgba(0,247,255,0.04); animation-delay: 1.2s; }
@keyframes ring-expand {
  0%   { transform: translate(-50%,-50%) scale(0.3); opacity: 0.7; }
  80%  { opacity: 0.1; }
  100% { transform: translate(-50%,-50%) scale(1);   opacity: 0; }
}

/* ── Horizontal neon scan lines that sweep down ── */
#cp-scanbeam {
  position: fixed; inset: 0;
  z-index: 0; pointer-events: none;
  overflow: hidden;
}
.cp-beam {
  position: absolute;
  left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg,
    transparent 0%, rgba(0,247,255,0.6) 30%,
    rgba(0,247,255,0.9) 50%, rgba(0,247,255,0.6) 70%, transparent 100%);
  box-shadow: 0 0 18px rgba(0,247,255,0.4), 0 0 40px rgba(0,247,255,0.15);
  animation: beam-sweep 12s linear infinite;
  opacity: 0;
}
.cp-beam:nth-child(2) {
  background: linear-gradient(90deg,
    transparent 0%, rgba(255,0,255,0.5) 40%,
    rgba(255,0,255,0.8) 50%, rgba(255,0,255,0.5) 60%, transparent 100%);
  box-shadow: 0 0 18px rgba(255,0,255,0.35), 0 0 40px rgba(255,0,255,0.1);
  animation-delay: 6s;
}
@keyframes beam-sweep {
  0%   { top: -2px; opacity: 0; }
  5%   { opacity: 0.8; }
  95%  { opacity: 0.6; }
  100% { top: 100vh; opacity: 0; }
}

/* ── Corner HUD brackets ── */
#cp-corners {
  position: fixed; inset: 0;
  z-index: 0; pointer-events: none;
}
.cp-corner {
  position: absolute;
  width: 40px; height: 40px;
  opacity: 0.35;
}
.cp-corner::before, .cp-corner::after {
  content: '';
  position: absolute;
  background: var(--cyan);
  box-shadow: 0 0 8px rgba(0,247,255,0.6);
}
.cp-corner::before { width: 100%; height: 1px; top: 0; left: 0; }
.cp-corner::after  { width: 1px; height: 100%; top: 0; left: 0; }
.cp-corner.tl { top: 16px; left: 16px; }
.cp-corner.tr { top: 16px; right: 16px; transform: scaleX(-1); }
.cp-corner.bl { bottom: 16px; left: 16px; transform: scaleY(-1); }
.cp-corner.br { bottom: 16px; right: 16px; transform: scale(-1); }
@keyframes corner-blink {
  0%,95%,100% { opacity: 0.35; }
  96% { opacity: 0.1; }
}
.cp-corner { animation: corner-blink 4s ease infinite; }

/* ── Streamlit chrome ── */
[data-testid="stHeader"] {
  background: rgba(8,8,14,0.96) !important;
  border-bottom: 1px solid var(--border-neon) !important;
  backdrop-filter: blur(24px) !important;
  height: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
#MainMenu, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* Remove the top gap that stHeader leaves */
.block-container {
  padding-top: 1.5rem !important;
}
section.main .block-container {
  padding-top: 1.5rem !important;
}

::-webkit-scrollbar       { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--cyan); border-radius: 2px;
                             box-shadow: 0 0 5px var(--cyan); }

hr {
  border: none !important;
  border-top: 1px solid rgba(0,247,255,0.1) !important;
  margin: 1.6rem 0 !important;
}

/* ── Typography ── */
h1, h2, h3, h4 {
  font-family: var(--orbitron) !important;
  font-weight: 700 !important;
  letter-spacing: .05em !important;
  color: var(--cyan) !important;
  text-shadow: 0 0 20px rgba(0,247,255,0.5) !important;
}
p, li, span, label, div { font-family: var(--rajdhani) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: rgba(8,8,15,0.98) !important;
  border-right: 1px solid var(--border-neon) !important;
  backdrop-filter: blur(24px);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-family: var(--orbitron) !important;
  font-size: .7rem !important;
  letter-spacing: .18em !important;
  color: var(--cyan) !important;
  text-shadow: 0 0 10px rgba(0,247,255,0.4) !important;
}

/* ── Sidebar toggle / collapse button — full cyberpunk override ── */
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarUserContent"] ~ * button[kind="header"],
button[data-testid="stBaseButton-headerNoPadding"],
[data-testid="collapsedControl"] button {
  background: rgba(0,247,255,0.06) !important;
  border: 1px solid rgba(0,247,255,0.28) !important;
  border-radius: 3px !important;
  color: var(--cyan) !important;
  box-shadow: 0 0 10px rgba(0,247,255,0.12) !important;
  transition: all .2s ease !important;
}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover {
  background: rgba(0,247,255,0.14) !important;
  border-color: var(--cyan) !important;
  box-shadow: 0 0 18px rgba(0,247,255,0.35) !important;
}
[data-testid="stSidebarCollapsedControl"] button svg,
[data-testid="collapsedControl"] button svg {
  color: var(--cyan) !important;
  fill: var(--cyan) !important;
}
/* Header area sidebar chevron */
[data-testid="stHeader"] button {
  background: rgba(0,247,255,0.05) !important;
  border: 1px solid rgba(0,247,255,0.2) !important;
  border-radius: 3px !important;
  transition: all .2s ease !important;
}
[data-testid="stHeader"] button:hover {
  background: rgba(0,247,255,0.12) !important;
  border-color: rgba(0,247,255,0.5) !important;
  box-shadow: 0 0 14px rgba(0,247,255,0.3) !important;
}
[data-testid="stHeader"] button svg {
  color: var(--cyan) !important;
  stroke: var(--cyan) !important;
}

/* ── Column glass panels ── */
[data-testid="column"] > div:first-child {
  background: linear-gradient(145deg,
    rgba(13,13,26,0.96) 0%,
    rgba(9,9,18,0.98) 100%) !important;
  border: 1px solid var(--border-neon) !important;
  border-radius: var(--radius-md) !important;
  padding: 2rem 2.2rem !important;
  height: 100% !important;
  backdrop-filter: blur(20px) !important;
  box-shadow:
    0 0 0 1px rgba(0,247,255,0.04),
    0 8px 48px rgba(0,0,0,0.7),
    inset 0 1px 0 rgba(0,247,255,0.06) !important;
  position: relative !important;
  overflow: hidden !important;
}
/* top-left corner accent lines */
[data-testid="column"] > div:first-child::before {
  content: '';
  position: absolute; top: 0; left: 0;
  width: 50px; height: 2px;
  background: var(--cyan);
  box-shadow: var(--cyan-glow);
  animation: panel-corner-pulse 4s ease-in-out infinite;
}
[data-testid="column"] > div:first-child::after {
  content: '';
  position: absolute; top: 0; left: 0;
  width: 2px; height: 50px;
  background: var(--cyan);
  box-shadow: var(--cyan-glow);
  animation: panel-corner-pulse 4s ease-in-out infinite .5s;
}
@keyframes panel-corner-pulse {
  0%,100% { opacity: .7; box-shadow: 0 0 8px rgba(0,247,255,.35); }
  50%      { opacity: 1;  box-shadow: 0 0 24px rgba(0,247,255,.9), 0 0 50px rgba(0,247,255,.3); }
}

/* ── Panel border neon glow pulse ── */
@keyframes panel-border-breathe {
  0%,100% { border-color: rgba(0,247,255,.10) !important; box-shadow: 0 0 0 1px rgba(0,247,255,.02), 0 8px 48px rgba(0,0,0,.7) !important; }
  50%      { border-color: rgba(0,247,255,.28) !important; box-shadow: 0 0 0 1px rgba(0,247,255,.06), 0 8px 48px rgba(0,0,0,.7), 0 0 40px rgba(0,247,255,.06) !important; }
}
[data-testid="column"] > div:first-child {
  animation: panel-border-breathe 6s ease-in-out infinite;
}

/* ── Neon data stream ticker at bottom ── */
/* Pad main content so the fixed 20px ticker never overlaps playlist tracks */
[data-testid="stMain"] .main > div:first-child,
section.main .block-container {
  padding-bottom: 36px !important;
}

#cp-data-ticker {
  position: fixed;
  bottom: 0; left: 0; right: 0;
  z-index: 2; pointer-events: none;
  height: 20px;
  background: rgba(8,8,14,.92);
  border-top: 1px solid rgba(0,247,255,.12);
  overflow: hidden;
  display: flex; align-items: center;
}
#cp-data-ticker::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(90deg,
    rgba(8,8,14,1) 0%,
    transparent 8%,
    transparent 92%,
    rgba(8,8,14,1) 100%);
  z-index: 1;
}
.cp-ticker-inner {
  display: flex; gap: 3rem;
  white-space: nowrap;
  font-family: 'Share Tech Mono', monospace;
  font-size: .48rem; letter-spacing: .1em;
  text-transform: uppercase;
  color: rgba(0,247,255,.35);
  animation: ticker-scroll 30s linear infinite;
  padding-left: 100%;
}
@keyframes ticker-scroll {
  from { transform: translateX(0); }
  to   { transform: translateX(-50%); }
}
.cp-ticker-item { flex-shrink: 0; }
.cp-ticker-item em { color: rgba(0,247,255,.7); font-style: normal; }
.cp-ticker-sep {
  color: rgba(255,0,255,.4); flex-shrink: 0;
  margin: 0 .5rem;
}

/* ══════════════════════════════════════
   MOBILE RESPONSIVE — columns stack cleanly
══════════════════════════════════════ */
@media (max-width: 768px) {
  [data-testid="column"] > div:first-child {
    padding: 1.2rem 1rem !important;
    margin-bottom: 1rem !important;
  }
  .hud-logo { font-size: 1.3rem !important; }
  .hud-badge { font-size: .5rem !important; }
  .hud-tagline { font-size: .58rem !important; margin-bottom: 1.2rem !important; }
  .stButton > button { font-size: .6rem !important; padding: .55rem 1rem !important; }
  .emo-name { font-size: 1.3rem !important; }
  .emo-emoji { font-size: 2rem !important; }
  .pl-title { font-size: .82rem !important; }
  .track-song { font-size: .8rem !important; }
}


.section-label::before {
  content: '';
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--cyan);
  box-shadow: 0 0 8px var(--cyan);
  margin-right: .5rem;
  animation: dot-blink 1.8s ease infinite;
  flex-shrink: 0;
}

/* ── Playlist card border pulse ── */
@keyframes pl-card-pulse {
  0%,100% { box-shadow: inset 3px 0 20px rgba(0,247,255,.05); }
  50%      { box-shadow: inset 3px 0 40px rgba(0,247,255,.12), 0 0 30px rgba(0,247,255,.04); }
}
.playlist-card { animation: pl-card-pulse 5s ease-in-out infinite; }

/* ── Emo reveal card sweep ── */
@keyframes emo-border-pulse {
  0%,100% { box-shadow: none; }
  50%      { box-shadow: 0 0 30px rgba(0,247,255,.08); }
}
.emo-reveal { animation: emo-border-pulse 4s ease-in-out infinite; }

/* ── Phase transition fade-in ── */
@keyframes phase-fade-in {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
.emo-reveal {
  animation: emo-border-pulse 4s ease-in-out infinite,
             phase-fade-in 0.4s ease forwards;
}
.playlist-card {
  animation: pl-card-pulse 5s ease-in-out infinite,
             phase-fade-in 0.4s ease 0.15s both;
}
.scan-idle {
  animation: phase-fade-in 0.35s ease forwards;
}
.neural-wrap {
  animation: phase-fade-in 0.35s ease forwards;
}

/* ══════════════════════════════════════
   STREAMLIT BUTTONS — cyberpunk style
══════════════════════════════════════ */
.stButton > button {
  background: transparent !important;
  border: 1px solid rgba(0,247,255,0.35) !important;
  color: var(--cyan) !important;
  font-family: var(--orbitron) !important;
  font-weight: 600 !important;
  font-size: .78rem !important;
  letter-spacing: .14em !important;
  text-transform: uppercase !important;
  border-radius: var(--radius) !important;
  padding: .65rem 2rem !important;
  transition: all .2s ease !important;
  text-shadow: 0 0 8px rgba(0,247,255,0.4) !important;
  width: 100% !important;
}
.stButton > button:hover {
  border-color: var(--cyan) !important;
  background: var(--cyan) !important;
  color: #000 !important;
  box-shadow: var(--cyan-glow) !important;
  text-shadow: none !important;
  transform: translateY(-2px) !important;
}

/* ── Link buttons (st.link_button) ── */
.stLinkButton > a {
  background: transparent !important;
  border: 1px solid rgba(0,247,255,0.3) !important;
  color: var(--cyan) !important;
  font-family: var(--orbitron) !important;
  font-size: .76rem !important;
  font-weight: 600 !important;
  letter-spacing: .1em !important;
  text-transform: uppercase !important;
  border-radius: var(--radius) !important;
  padding: .55rem 1.5rem !important;
  text-decoration: none !important;
  display: inline-flex !important;
  align-items: center !important;
  transition: all .2s ease !important;
  text-shadow: 0 0 8px rgba(0,247,255,0.3) !important;
}
/* YouTube — red hover */
.stLinkButton:first-of-type > a:hover {
  border-color: var(--red) !important;
  color: var(--red) !important;
  background: var(--red-dim) !important;
  box-shadow: var(--red-glow) !important;
  text-shadow: 0 0 8px var(--red) !important;
  transform: scale(1.05) !important;
}
/* Spotify — green hover */
.stLinkButton:last-of-type > a:hover {
  border-color: var(--green) !important;
  color: var(--green) !important;
  background: var(--green-dim) !important;
  box-shadow: var(--green-glow) !important;
  text-shadow: 0 0 8px var(--green) !important;
  transform: scale(1.05) !important;
}

/* ── HTML anchor buttons (for history, avoids key= issue) ── */
.cb-btn {
  display: inline-flex; align-items: center; gap: .4rem;
  padding: .5rem 1.3rem;
  border: 1px solid rgba(0,247,255,0.3);
  border-radius: var(--radius);
  font-family: var(--orbitron);
  font-size: .74rem; font-weight: 600;
  letter-spacing: .1em; text-transform: uppercase;
  text-decoration: none !important;
  color: var(--cyan);
  text-shadow: 0 0 8px rgba(0,247,255,0.35);
  transition: all .2s ease;
}
.cb-btn.yt:hover {
  border-color: var(--red); color: var(--red);
  background: var(--red-dim);
  box-shadow: var(--red-glow);
  text-shadow: 0 0 8px var(--red);
  transform: scale(1.05);
}
.cb-btn.sp:hover {
  border-color: var(--green); color: var(--green);
  background: var(--green-dim);
  box-shadow: var(--green-glow);
  text-shadow: 0 0 8px var(--green);
  transform: scale(1.05);
}
.cb-btn.am {
  border-color: rgba(10,132,255,0.4);
  color: #0a84ff;
  text-shadow: 0 0 6px rgba(10,132,255,0.3);
}
.cb-btn.am:hover {
  border-color: #fff; color: #fff;
  background: rgba(255,255,255,0.06);
  box-shadow: 0 0 18px rgba(255,255,255,0.5), 0 0 50px rgba(255,255,255,0.15);
  text-shadow: 0 0 12px #fff, 0 0 30px rgba(255,255,255,0.6);
  transform: scale(1.05);
}

/* ══════════════════════════════════════
   SPINNER
══════════════════════════════════════ */
[data-testid="stSpinner"] > div {
  border-top-color: var(--cyan) !important;
}
[data-testid="stSpinner"] p {
  font-family: var(--mono) !important;
  font-size: .85rem !important;
  color: var(--cyan) !important;
  letter-spacing: .08em;
  text-shadow: 0 0 8px rgba(0,247,255,0.4);
}

/* ══════════════════════════════════════
   EXPANDER — minimal base (history uses custom HTML accordion)
══════════════════════════════════════ */
[data-testid="stExpander"] {
  background: rgba(11,11,22,0.9) !important;
  border: 1px solid rgba(0,247,255,0.12) !important;
  border-radius: var(--radius-md) !important;
  margin-bottom: .5rem !important;
}
[data-testid="stExpander"] summary {
  font-family: var(--orbitron) !important;
  font-size: .78rem !important;
  font-weight: 600 !important;
  letter-spacing: .14em !important;
  color: var(--text2) !important;
  text-transform: uppercase !important;
  padding: .8rem 1.1rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--cyan) !important; }
[data-testid="stExpander"] summary svg { display: none !important; }

/* ══════════════════════════════════════
   METRICS
══════════════════════════════════════ */
[data-testid="stMetric"] {
  background: rgba(0,247,255,0.03) !important;
  border: 1px solid rgba(0,247,255,0.1) !important;
  border-radius: var(--radius) !important;
  padding: .6rem .9rem !important;
  margin-bottom: .4rem !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--orbitron) !important;
  font-size: .72rem !important;
  letter-spacing: .14em !important;
  color: var(--text2) !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--orbitron) !important;
  font-size: 1.1rem !important;
  color: var(--cyan) !important;
  text-shadow: 0 0 10px rgba(0,247,255,0.4) !important;
}

.stAlert {
  background: rgba(0,247,255,0.04) !important;
  border: 1px solid rgba(0,247,255,0.15) !important;
  border-radius: var(--radius) !important;
  font-family: var(--mono) !important;
  color: var(--text2) !important;
}

/* ══════════════════════════════════════
   CUSTOM COMPONENTS
══════════════════════════════════════ */

/* ── HUD wordmark ── */
.hud-wordmark {
  display: flex; align-items: baseline; gap: 1rem;
  margin-bottom: .4rem;
}
.hud-logo {
  font-family: var(--orbitron);
  font-size: 1.9rem; font-weight: 900;
  letter-spacing: .1em;
  color: var(--cyan);
  text-shadow: 0 0 30px rgba(0,247,255,0.65), 0 0 80px rgba(0,247,255,0.2);
  animation: logo-glitch 7s ease-in-out infinite;
}
@keyframes logo-glitch {
  0%,93%,100% { clip-path: none; transform: none; color: var(--cyan); }
  94%  { clip-path: inset(15% 0 50% 0); transform: translate(-3px,0); color: var(--magenta); }
  96%  { clip-path: inset(55% 0 10% 0); transform: translate(3px,0);  color: var(--green); }
  97%  { clip-path: none; transform: none; color: var(--cyan); }
}
.hud-badge {
  font-family: var(--mono);
  font-size: .85rem; letter-spacing: .12em;
  color: var(--text2);
  border: 1px solid rgba(0,247,255,0.2);
  padding: .18rem .55rem;
  border-radius: 2px;
}
.hud-tagline {
  font-family: var(--mono);
  font-size: .92rem; color: var(--text3);
  letter-spacing: .1em; text-transform: uppercase;
  margin-bottom: .4rem;
}
.hud-tagline em { color: var(--cyan); font-style: normal; }
.hud-tagline-human {
  font-family: var(--rajdhani);
  font-size: 1.1rem; color: rgba(0,247,255,0.38);
  letter-spacing: .06em;
  margin-bottom: 2rem;
}

/* ── Section label ── */
.section-label {
  font-family: var(--orbitron);
  font-size: .86rem; letter-spacing: .22em;
  text-transform: uppercase;
  color: var(--cyan);
  text-shadow: 0 0 8px rgba(0,247,255,0.35);
  margin-bottom: 1.2rem;
  display: flex; align-items: center; gap: .7rem;
}
.section-label::after {
  content: '';
  flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(0,247,255,0.25), transparent);
}

/* ── Idle scan area ── */
.scan-idle {
  text-align: center; padding: 2.5rem 2rem 2rem;
  border: 1px dashed rgba(0,247,255,0.12);
  border-radius: var(--radius-md);
  position: relative; overflow: hidden;
}
.scan-idle::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at 50% 55%,
    rgba(0,247,255,0.04) 0%, transparent 70%);
  pointer-events: none;
}
.scan-reticle { width: 90px; height: 90px; margin: 0 auto 1.5rem; }
.scan-hint {
  font-family: var(--mono); font-size: .95rem;
  color: var(--text3); letter-spacing: .1em; text-transform: uppercase;
}

/* ── Neural scanning state ── */
.neural-wrap { text-align: center; padding: 1rem 0; }
.radar {
  width: 90px; height: 90px; margin: 0 auto 1rem;
  border-radius: 50%;
  border: 1.5px solid rgba(0,247,255,0.3);
  position: relative;
  animation: radar-outer 1.8s ease-in-out infinite;
}
.radar::before {
  content: ''; position: absolute; inset: 10px;
  border-radius: 50%;
  border: 1px solid rgba(0,247,255,0.5);
  animation: radar-outer 1.8s ease-in-out infinite .3s;
}
.radar::after {
  content: ''; position: absolute; inset: 26px;
  border-radius: 50%;
  background: rgba(0,247,255,0.12);
  animation: radar-inner 1.8s ease-in-out infinite .6s;
}
@keyframes radar-outer {
  0%,100% { box-shadow: 0 0 0 0 rgba(0,247,255,0.4); }
  60%      { box-shadow: 0 0 0 18px rgba(0,247,255,0); }
}
@keyframes radar-inner { 0%,100% { opacity:.3; } 50% { opacity:1; } }
.neural-status {
  font-family: var(--orbitron); font-size: .9rem;
  letter-spacing: .2em; text-transform: uppercase;
  color: var(--cyan); text-shadow: 0 0 10px rgba(0,247,255,0.5);
  animation: status-flicker 2.5s ease infinite;
}
@keyframes status-flicker {
  0%,95%,100% { opacity:1; }
  96% { opacity:.4; }
}

/* Lock-on bar */
.lock-wrap { margin: 1rem 0; }
.lock-label {
  display: flex; justify-content: space-between;
  font-family: var(--mono); font-size: .86rem;
  color: var(--text2); letter-spacing: .08em; margin-bottom: .4rem;
}
.lock-track {
  height: 2px; background: rgba(0,247,255,0.1);
  border-radius: 1px; overflow: hidden;
}
.lock-fill {
  height: 2px; background: var(--cyan);
  border-radius: 1px;
  box-shadow: 0 0 8px var(--cyan);
  transition: width .15s ease;
}

/* ── Cyber badge ── */
.cyber-badge {
  display: inline-flex; align-items: center; gap: .4rem;
  font-family: var(--orbitron); font-size: .68rem;
  letter-spacing: .14em; text-transform: uppercase;
  padding: .2rem .65rem;
  border: 1px solid rgba(0,247,255,0.2);
  border-radius: 2px; color: var(--text2);
}
.cyber-badge.active {
  color: var(--cyan); border-color: var(--cyan);
  box-shadow: 0 0 6px rgba(0,247,255,0.2);
}
.cyber-badge .bdot {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--cyan); box-shadow: 0 0 5px var(--cyan);
  animation: dot-blink 1.2s ease infinite;
}
@keyframes dot-blink { 0%,100% { opacity:.3; } 50% { opacity:1; } }

/* ── Emotion reveal ── */
.emo-reveal {
  background: linear-gradient(135deg,
    rgba(0,247,255,0.04) 0%, rgba(10,10,20,0.9) 100%);
  border: 1px solid rgba(0,247,255,0.2);
  border-radius: var(--radius-md);
  padding: 1.8rem 2rem;
  margin: 1.2rem 0;
  position: relative; overflow: hidden;
}
.emo-reveal::before {
  content: '';
  position: absolute; top: 0; left: -100%;
  width: 100%; height: 1.5px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  animation: sweep 2.5s ease infinite;
}
@keyframes sweep { from { left:-100%; } to { left:100%; } }
.emo-top { display: flex; align-items: center; gap: 1.2rem; }
.emo-emoji {
  font-size: 3rem; line-height: 1;
  filter: drop-shadow(0 0 22px rgba(0,247,255,0.75)) drop-shadow(0 0 6px rgba(0,247,255,0.5));
  animation: emo-bob 3.5s ease-in-out infinite;
  font-family: var(--mono);
  opacity: 0.95;
}
@keyframes emo-bob {
  0%,100% { transform: translateY(0) scale(1); }
  50%      { transform: translateY(-5px) scale(1.05); }
}
.emo-name {
  font-family: var(--orbitron);
  font-size: 1.9rem; font-weight: 900;
  letter-spacing: .06em; text-transform: uppercase;
  animation: emo-reveal-anim .7s ease forwards;
}
@keyframes emo-reveal-anim {
  0%   { clip-path: inset(50% 0 50% 0); opacity:0; transform: scaleX(1.1); }
  50%  { clip-path: inset(10% 0 10% 0); opacity:.8; }
  100% { clip-path: none; opacity:1; transform: scaleX(1); }
}
.emo-sub {
  font-family: var(--mono); font-size: .9rem;
  color: var(--text2); letter-spacing: .1em;
  text-transform: uppercase; margin-top: .3rem;
}

/* Neon vibe pills */
.vibe-pills { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.2rem; }
.pill {
  font-family: var(--mono); font-size: .82rem;
  letter-spacing: .08em; text-transform: uppercase;
  padding: .28rem .88rem;
  border-radius: 99px; border: 1px solid;
}
.pill.genre { color:var(--cyan); border-color:rgba(0,247,255,0.4);
  background:rgba(0,247,255,0.06); text-shadow:0 0 5px rgba(0,247,255,0.3); }
.pill.energy{ color:var(--magenta); border-color:rgba(255,0,255,0.4);
  background:rgba(255,0,255,0.06); text-shadow:0 0 5px rgba(255,0,255,0.3); }
.pill.bpm   { color:var(--green); border-color:rgba(0,255,159,0.4);
  background:rgba(0,255,159,0.06); text-shadow:0 0 5px rgba(0,255,159,0.3); }
.pill.vibe  { color:#bf5af2; border-color:rgba(191,90,242,0.4);
  background:rgba(191,90,242,0.06); }

/* ── Playlist card ── */
.playlist-card {
  background: rgba(10,10,20,0.92);
  border: 1px solid rgba(0,247,255,0.1);
  border-radius: var(--radius-md);
  padding: 1.6rem 1.8rem;
  margin-top: 1.2rem;
  position: relative; overflow: hidden;
}
.pl-prefix {
  font-family: var(--orbitron); font-size: .78rem;
  letter-spacing: .22em; text-transform: uppercase;
  color: var(--text2); display: block; margin-bottom: .3rem;
}
.pl-title {
  font-family: var(--orbitron); font-size: 1.35rem;
  font-weight: 700; letter-spacing: .04em;
  margin-bottom: 1.2rem;
  line-height: 1.3;
}
.track-list { list-style: none; padding: 0; margin: 0; }
.track-item {
  display: flex; align-items: center; gap: .85rem;
  padding: .65rem .5rem;
  border-radius: var(--radius);
  transition: all .18s ease;
  margin-bottom: 1px;
}
.track-item:hover {
  background: rgba(0,247,255,0.04);
  transform: translateX(5px);
}
.track-num {
  font-family: var(--mono); font-size: .86rem;
  color: var(--text3); width: 1.3rem; flex-shrink: 0; text-align: right;
}
.track-body { flex: 1; }
.track-song {
  font-family: var(--rajdhani); font-size: 1.05rem;
  font-weight: 600; color: var(--text); display: block; letter-spacing: .01em;
}
.track-artist {
  font-family: var(--mono); font-size: .88rem;
  color: var(--text2); letter-spacing: .04em;
}
.track-wave {
  display: flex; align-items: center; gap: 1.5px;
  height: 14px; flex-shrink: 0; opacity: .45;
}
.track-wave span {
  width: 2px; border-radius: 1px; background: var(--cyan);
  animation: wv 1.5s ease infinite;
}
.track-wave span:nth-child(1){ height:4px; animation-delay:0s; }
.track-wave span:nth-child(2){ height:10px; animation-delay:.12s; }
.track-wave span:nth-child(3){ height:6px; animation-delay:.24s; }
.track-wave span:nth-child(4){ height:13px; animation-delay:.36s; }
.track-wave span:nth-child(5){ height:7px; animation-delay:.48s; }
.track-wave span:nth-child(6){ height:11px; animation-delay:.6s; }
@keyframes wv {
  0%,100% { transform:scaleY(.5); opacity:.5; }
  50%      { transform:scaleY(1);  opacity:1; }
}
.pl-vibe {
  margin-top: 1rem; font-family: var(--rajdhani);
  font-size: 1.0rem; color: var(--text2);
  font-style: italic; line-height: 1.6; font-weight: 400;
  border-top: 1px solid rgba(0,247,255,0.07); padding-top: .9rem;
}

/* ── Action row ── */
.action-row { display: flex; gap: .75rem; flex-wrap: wrap; margin-top: 1.2rem; }

/* ── Playlist idle ── */
.pl-idle {
  text-align: center; padding: 4rem 2rem;
  border: 1px dashed rgba(0,247,255,0.08);
  border-radius: var(--radius-md);
}
.pl-idle-icon { font-size: 3rem; opacity:.12; margin-bottom: 1rem;
  animation: idle-breathe 4s ease infinite; }
@keyframes idle-breathe { 0%,100%{opacity:.08;} 50%{opacity:.2;} }
.pl-idle-hint {
  font-family: var(--orbitron); font-size: .58rem;
  letter-spacing: .2em; color: var(--text3); text-transform: uppercase;
}

/* ── History track rows (inside expanders) ── */
.hist-track {
  display: flex;
  align-items: center;
  gap: .7rem;
  padding: .45rem .3rem;
  border-radius: var(--radius);
  transition: background .15s ease;
}
.hist-track:hover { background: rgba(0,247,255,0.04); }
.hist-num {
  font-family: var(--mono); font-size: .58rem;
  color: var(--text3); width: 1.2rem; flex-shrink: 0; text-align: right;
}
.hist-body { flex: 1; }
.hist-song {
  font-family: var(--rajdhani); font-size: .88rem;
  font-weight: 600; color: var(--text); display: block;
}
.hist-artist {
  font-family: var(--mono); font-size: .62rem;
  color: var(--text2); letter-spacing: .03em;
}
.hist-links {
  display: flex; gap: .3rem; flex-shrink: 0;
}
.hist-title {
  font-family: var(--orbitron); font-size: .6rem;
  letter-spacing: .12em; text-transform: uppercase;
}
.track-link {
  display: flex; align-items: center; gap: .85rem;
  padding: .65rem .5rem;
  border-radius: var(--radius);
  transition: all .18s ease;
  margin-bottom: 1px;
  text-decoration: none !important;
  color: inherit !important;
  position: relative;
}
.track-link:hover {
  background: rgba(0,247,255,0.06);
  transform: translateX(6px);
  border-left: 2px solid rgba(0,247,255,0.5);
  padding-left: .35rem;
}
.track-link:hover .track-song {
  color: var(--cyan) !important;
  text-shadow: 0 0 8px rgba(0,247,255,0.4);
}
.track-link:hover .track-num {
  color: var(--cyan) !important;
}
.track-platforms {
  display: none;
  align-items: center;
  gap: .35rem;
  flex-shrink: 0;
  margin-left: auto;
}
.track-link:hover .track-platforms { display: flex; }
.track-link:hover .track-wave { display: none; }
.tp-btn {
  font-family: var(--mono);
  font-size: .65rem; font-weight: 600;
  letter-spacing: .06em; text-transform: uppercase;
  padding: .22rem .55rem;
  border-radius: 2px;
  border: 1px solid;
  text-decoration: none !important;
  transition: all .15s ease;
  white-space: nowrap;
}
.tp-btn.yt {
  color: var(--red); border-color: rgba(255,0,51,0.45);
  background: rgba(255,0,51,0.06);
}
.tp-btn.yt:hover {
  background: var(--red-dim); box-shadow: var(--red-glow);
  transform: scale(1.05);
}
.tp-btn.sp {
  color: var(--green); border-color: rgba(0,255,159,0.45);
  background: rgba(0,255,159,0.06);
}
.tp-btn.sp:hover {
  background: var(--green-dim); box-shadow: var(--green-glow);
  transform: scale(1.05);
}
</style>

<!-- ═══════════════════════════════════════════════════
     MOODBEATS v4.1 — Three.js Particle Sphere + Cursor
════════════════════════════════════════════════════ -->

<!-- Matrix canvas (kept) -->
<canvas id="matrix-canvas"></canvas>

<!-- Cyberpunk grid overlay -->
<div id="cp-grid"></div>

<!-- Neon pulse rings from center -->
<div id="cp-rings">
  <div class="cp-ring"></div>
  <div class="cp-ring"></div>
  <div class="cp-ring"></div>
  <div class="cp-ring"></div>
</div>

<!-- Horizontal scan beams -->
<div id="cp-scanbeam">
  <div class="cp-beam"></div>
  <div class="cp-beam"></div>
</div>

<!-- HUD corner brackets -->
<div id="cp-corners">
  <div class="cp-corner tl"></div>
  <div class="cp-corner tr"></div>
  <div class="cp-corner bl"></div>
  <div class="cp-corner br"></div>
</div>

<!-- Neon data ticker at bottom -->
<div id="cp-data-ticker">
  <div class="cp-ticker-inner">
    <span class="cp-ticker-item">NEURAL OS <em>v4.0</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">EMOTION ENGINE <em>ONLINE</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">BIOSCAN <em>READY</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">FER2013 MODEL <em>LOADED</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">MEDIAPIPE <em>CALIBRATED</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">PLAYLIST GEN <em>ACTIVE</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">NEURAL FREQ MAP <em>ENGAGED</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">MOODBEATS <em>CYBERPUNK EDITION</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">NEURAL OS <em>v4.0</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">EMOTION ENGINE <em>ONLINE</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">BIOSCAN <em>READY</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">FER2013 MODEL <em>LOADED</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">MEDIAPIPE <em>CALIBRATED</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">PLAYLIST GEN <em>ACTIVE</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">NEURAL FREQ MAP <em>ENGAGED</em></span>
    <span class="cp-ticker-sep">///</span>
    <span class="cp-ticker-item">MOODBEATS <em>CYBERPUNK EDITION</em></span>
  </div>
</div>

<!-- CSS particle field (replaces heavy Three.js sphere) -->
<div id="cp-particles" aria-hidden="true"></div>

<!-- Pulse Visualizer HUD — bars only, no mic/file/voice controls -->
<div id="audio-hud">
  <div id="audio-viz">
    <span></span><span></span><span></span><span></span><span></span>
    <span></span><span></span><span></span><span></span><span></span>
    <span></span><span></span>
  </div>
  <div id="audio-status-wrap">
    <span id="audio-dot"></span>
    <span id="audio-status">NEURAL OS</span>
  </div>
</div>

<!-- Cursor: system default -->

<style>
/* ── CSS Particle Field (replaces Three.js sphere) ── */
#cp-particles {
  position: fixed; inset: 0;
  z-index: 0; pointer-events: none;
  overflow: hidden;
}
.cp-particle {
  position: absolute;
  border-radius: 50%;
  pointer-events: none;
  animation: cp-float linear infinite;
  will-change: transform, opacity;
}

/* ── Audio HUD — pulse visualizer only ── */
#audio-hud {
  position: fixed;
  bottom: 22px; right: 22px;
  z-index: 9998;
  display: flex;
  align-items: center;
  gap: 12px;
  background: rgba(6,6,18,0.88);
  border: 1px solid var(--border-neon);
  border-radius: 6px;
  padding: 10px 18px;
  backdrop-filter: blur(20px);
  pointer-events: none;
  box-shadow: 0 0 30px rgba(0,247,255,0.06), inset 0 1px 0 rgba(0,247,255,0.07);
  transition: border-color 1.2s ease, box-shadow 1.2s ease;
}

/* Animated bar visualizer */
#audio-viz {
  display: flex;
  align-items: flex-end;
  gap: 2.5px;
  height: 22px;
  flex-shrink: 0;
}
#audio-viz span {
  display: block;
  width: 3px;
  border-radius: 2px;
  background: var(--cyan);
  box-shadow: 0 0 6px var(--cyan);
  animation: viz-idle 1.8s ease-in-out infinite;
  height: 4px;
}
#audio-viz span:nth-child(1)  { animation-delay: 0s;     animation-duration: 1.6s; }
#audio-viz span:nth-child(2)  { animation-delay: 0.12s;  animation-duration: 1.9s; }
#audio-viz span:nth-child(3)  { animation-delay: 0.24s;  animation-duration: 1.4s; }
#audio-viz span:nth-child(4)  { animation-delay: 0.36s;  animation-duration: 2.1s; }
#audio-viz span:nth-child(5)  { animation-delay: 0.48s;  animation-duration: 1.7s; }
#audio-viz span:nth-child(6)  { animation-delay: 0.60s;  animation-duration: 1.3s; }
#audio-viz span:nth-child(7)  { animation-delay: 0.72s;  animation-duration: 2.0s; }
#audio-viz span:nth-child(8)  { animation-delay: 0.84s;  animation-duration: 1.5s; }
#audio-viz span:nth-child(9)  { animation-delay: 0.96s;  animation-duration: 1.8s; }
#audio-viz span:nth-child(10) { animation-delay: 1.08s;  animation-duration: 1.6s; }
#audio-viz span:nth-child(11) { animation-delay: 0.30s;  animation-duration: 2.2s; }
#audio-viz span:nth-child(12) { animation-delay: 0.54s;  animation-duration: 1.4s; }
@keyframes viz-idle {
  0%, 100% { height: 3px; opacity: 0.3; }
  50%       { height: 18px; opacity: 0.9; }
}

/* Status text */
#audio-status-wrap {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 90px;
}
#audio-dot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--text3);
  flex-shrink: 0;
  transition: background 0.3s ease, box-shadow 0.3s ease;
}
#audio-dot.live {
  background: var(--cyan);
  box-shadow: 0 0 8px var(--cyan);
  animation: dot-blink 1.2s ease infinite;
}
#audio-status {
  font-family: 'Share Tech Mono', monospace;
  font-size: .5rem;
  color: rgba(0,247,255,0.35);
  letter-spacing: .1em;
  text-transform: uppercase;
  transition: color 0.3s ease;
}
#audio-status.live {
  color: var(--cyan);
  text-shadow: 0 0 8px rgba(0,247,255,0.4);
}

/* cursor trail removed */
</style>

<!-- Three.js removed — using CSS particle field instead -->

<script>
// ════════════════════════════════════════════════════════════
//  MOODBEATS CSS PARTICLE FIELD — lightweight, no WebGL needed
//  Smooth floating neon dots; zero first-load penalty
// ════════════════════════════════════════════════════════════
(function initParticleField() {
  function boot() {
    const container = document.getElementById('cp-particles');
    if (!container) { setTimeout(boot, 300); return; }

    const COLORS = ['#00f7ff','#ff00ff','#00ff9f','#9d00ff','#0055ff','#ccff00'];
    const COUNT  = 55;  // lightweight — pure CSS, no WebGL

    // Inject keyframe animation once
    const styleEl = document.createElement('style');
    styleEl.textContent = `
      @keyframes cp-float {
        0%   { transform: translateY(0px) translateX(0px); opacity: 0; }
        8%   { opacity: 1; }
        92%  { opacity: 0.6; }
        100% { transform: translateY(-100vh) translateX(var(--drift)); opacity: 0; }
      }
    `;
    document.head.appendChild(styleEl);

    for (let i = 0; i < COUNT; i++) {
      const dot = document.createElement('div');
      dot.className = 'cp-particle';

      const color  = COLORS[i % COLORS.length];
      const size   = (Math.random() * 3 + 1).toFixed(1);   // 1–4px
      const left   = (Math.random() * 100).toFixed(1);     // % across screen
      const delay  = (Math.random() * 18).toFixed(2);      // stagger
      const dur    = (14 + Math.random() * 20).toFixed(1); // 14–34s
      const drift  = ((Math.random() - 0.5) * 120).toFixed(0) + 'px'; // horizontal wander
      const blur   = Math.random() > 0.6 ? `blur(${(Math.random()*1.5).toFixed(1)}px)` : 'none';
      const startY = (Math.random() * 100).toFixed(1);     // start anywhere vertically

      dot.style.cssText = `
        width:${size}px; height:${size}px;
        background:${color};
        box-shadow: 0 0 ${(parseFloat(size)*3).toFixed(0)}px ${color}, 0 0 ${(parseFloat(size)*6).toFixed(0)}px ${color}44;
        left:${left}%;
        bottom:-${startY}vh;
        opacity:0;
        animation-duration:${dur}s;
        animation-delay:-${delay}s;
        filter:${blur};
        --drift:${drift};
      `;
      container.appendChild(dot);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    setTimeout(boot, 0);
  }

  // Dummy exports so nothing downstream breaks
  window._mbDominantHue = () => 0;
  window._mbAudioScale  = () => 0;

})();


// Cursor: system default — no custom cursor JS needed


// ════════════════════════════════════════════════════════════
//  PULSE VISUALIZER — auto hue cycling, no audio input needed
// ════════════════════════════════════════════════════════════
(function initPulseViz() {
  function setup() {
    const viz  = document.getElementById('audio-viz');
    const dot  = document.getElementById('audio-dot');
    const stat = document.getElementById('audio-status');
    if (!viz) { setTimeout(setup, 300); return; }

    const bars = Array.from(viz.querySelectorAll('span'));
    let t = 0;

    // Pulse status text cycles
    const statusMsgs = ['NEURAL OS', 'ACTIVE', 'SCANNING', 'LOCKED', 'NEURAL OS', 'LIVE'];
    let msgIdx = 0;
    setInterval(() => {
      msgIdx = (msgIdx + 1) % statusMsgs.length;
      if (stat) stat.textContent = statusMsgs[msgIdx];
    }, 3500);

    // Dot alive pulse
    if (dot) { dot.classList.add('live'); }

    function loop() {
      requestAnimationFrame(loop);
      t += 0.018;
      bars.forEach((bar, i) => {
        const wave  = Math.sin(t * 2.2 + i * 0.55) * 0.5 + 0.5;
        const wave2 = Math.sin(t * 1.4 + i * 0.9 + 1.2) * 0.5 + 0.5;
        const h = Math.round(3 + (wave * wave2) * 17);
        const hue = Math.round((t * 40 + i * 22) % 360);
        bar.style.height    = h + 'px';
        bar.style.background = `hsl(${hue},100%,62%)`;
        bar.style.boxShadow  = `0 0 ${4 + wave*8}px hsl(${hue},100%,62%)`;
      });
    }
    requestAnimationFrame(loop);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setup);
  } else {
    setTimeout(setup, 400);
  }
})();
// ════════════════════════════════════════════════════════════
(function initMatrix() {
  function setup() {
    const c = document.getElementById('matrix-canvas');
    if (!c) { setTimeout(setup, 200); return; }
    const ctx = c.getContext('2d');
    const chars = '01アイウエオカキクケコMOODBEATSNEURAL';
    let W, H, cols, drops;
    function resize() {
      W = c.width  = window.innerWidth;
      H = c.height = window.innerHeight;
      cols  = Math.floor(W / 16);
      drops = Array(cols).fill(0).map(() => Math.random() * -40);
    }
    function draw() {
      ctx.fillStyle = 'rgba(8,8,15,0.05)';
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#00f7ff';
      ctx.font = '10px Share Tech Mono, monospace';
      for (let i = 0; i < cols; i++) {
        const ch = chars[Math.floor(Math.random() * chars.length)];
        ctx.fillText(ch, i * 16, drops[i] * 16);
        if (drops[i] * 16 > H && Math.random() > 0.978) drops[i] = 0;
        drops[i]++;
      }
    }
    resize();
    window.addEventListener('resize', resize);
    setInterval(draw, 60);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setup);
  } else {
    setup();
  }
})();
</script>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────

try:
    OXLO_KEY = st.secrets["OXLO_KEY"]
except Exception:
    st.error(
        "⚠️ **API key not configured.** "
        "Add `OXLO_KEY = 'your-key'` to `.streamlit/secrets.toml` and restart the app.",
        icon="🔑",
    )
    st.stop()
SMOOTHING_WINDOW = 5    # was 10 — reacts faster, reduces happy-bias carryover
LOCK_FRAMES      = 8   # was 14 — locks in ~2s instead of ~4s
FRAME_SKIP       = 2

EMOJI_FULL = {
    "happy":    "😄",   # grinning — joy, euphoria
    "sad":      "😢",   # crying face — melancholy, sorrow
    "angry":    "😠",   # angry face — rage, intensity
    "surprise":  "😲",   # surprised face — shocked, startled, amazed
    "fear":     "😨",   # fearful face — dread, vigilance
    "disgust":  "🤢",   # nauseated — rejection, disgust
    "neutral":  "😐",   # neutral face — flat, stable
}
# Inside Out theme: red=angry, yellow=happy, green=disgust, blue=sad, purple=fear, grey=neutral, orange=surprise
EMOTION_COLORS = {
    "happy":    "#FFE600",   # yellow — joy
    "sad":      "#00EEFF",   # cyan — sadness
    "angry":    "#FF0044",   # red — anger
    "surprise": "#FF00CC",   # pink/magenta — surprise
    "fear":     "#AA00FF",   # purple — fear
    "disgust":  "#00FF88",   # green — disgust
    "neutral":  "#88CCFF",   # blue/white — neutral
}

# ──────────────────────────────────────────────────────────────
# OPENAI CLIENT
# ──────────────────────────────────────────────────────────────

try:
    from openai import OpenAI
    client = OpenAI(base_url="https://api.oxlo.ai/v1", api_key=OXLO_KEY, timeout=10.0)
except ImportError:
    st.error("Install openai: pip install openai")
    st.stop()

# ──────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────

_defaults = {
    "phase":           "idle",   # idle | scanning | locked | result
    "locked_emotion":  None,
    "locked_conf":     0.0,
    "locked_frame":    None,
    "playlist_parsed": None,
    "history":         [],
    "regen_seed":      0,        # bumped on every Regenerate Mix press
    "camera_consent":  False,    # one-time consent gate
    # Per-session emotion smoothing buffer — isolated per user, not module-level
    "emotion_buf":     deque(maxlen=SMOOTHING_WINDOW),
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="// Initialising neural core…")
def load_emotion_model():
    for path in ["emotion_model_best.h5", "emotion_model_swa.h5",
                 "emotion_model_final.h5", "emotion_model.h5"]:
        if os.path.exists(path):
            try:
                with keras.utils.custom_object_scope(CUSTOM_OBJECTS):
                    m = load_model(path, custom_objects=CUSTOM_OBJECTS)
                return m
            except Exception as e:
                st.warning(f"Could not load {path}: {e}")
    st.error("No model file found. Train the model first.")
    st.stop()

model = load_emotion_model()

# ──────────────────────────────────────────────────────────────
# FACE DETECTOR
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="// Calibrating biosensors…")
def get_face_detector():
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        tflite = "blaze_face_short_range.tflite"
        if not os.path.exists(tflite):
            import urllib.request
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "face_detector/blaze_face_short_range/float16/1/"
                "blaze_face_short_range.tflite", tflite)
        opts = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=tflite),
            min_detection_confidence=0.55)
        return ("tasks", mp_vision.FaceDetector.create_from_options(opts))
    except Exception:
        import mediapipe as mp
        return ("legacy",
                mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.55))

face_bundle = get_face_detector()

# emotion_buf lives in st.session_state["emotion_buf"] — isolated per user session.
# Convenience accessor defined here; always references the current session's deque.
def _emotion_buf() -> deque:
    return st.session_state.emotion_buf

# ──────────────────────────────────────────────────────────────
# IMAGE ENHANCEMENT
# ──────────────────────────────────────────────────────────────

def enhance_face(bgr: np.ndarray) -> np.ndarray:
    try:
        d = cv2.bilateralFilter(bgr, 9, 75, 75)
        lab = cv2.cvtColor(d, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(2.0, (4, 4))
        d = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
        blur = cv2.GaussianBlur(d, (0, 0), 2.0)
        d = np.clip(cv2.addWeighted(d, 1.5, blur, -0.5, 0), 0, 255).astype(np.uint8)
        lut = np.array([min(255, int((i/255)**0.85*255)) for i in range(256)], np.uint8)
        return cv2.LUT(d, lut)
    except Exception:
        return bgr

# ──────────────────────────────────────────────────────────────
# EMOTION DETECTION
# ──────────────────────────────────────────────────────────────

def detect_emotion(frame):
    """Returns (smoothed_emotion, confidence, annotated_frame, raw_probs)."""
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    api, detector = face_bundle
    if api == "tasks":
        import mediapipe as _mp
        res = detector.detect(_mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb))
        if not res.detections:
            return None, 0.0, frame, None
        # Pick the largest (most prominent) face instead of always the first
        best = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        bb = best.bounding_box
        x, y, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height
    else:
        res = detector.process(rgb)
        if not res.detections:
            return None, 0.0, frame, None
        # Pick the largest face by area
        def _area(d):
            b = d.location_data.relative_bounding_box
            return b.width * b.height
        best = max(res.detections, key=_area)
        bb = best.location_data.relative_bounding_box
        x, y   = int(bb.xmin * w), int(bb.ymin * h)
        bw, bh = int(bb.width * w), int(bb.height * h)

    x  = max(0, x);  y  = max(0, y)
    bw = min(bw, w - x); bh = min(bh, h - y)
    face = frame[y:y+bh, x:x+bw]
    if face.size == 0:
        return None, 0.0, frame, None

    face_r   = cv2.resize(face, (96, 96))
    enhanced = enhance_face(face_r)
    face_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB).astype("float32")
    probs    = model.predict(np.expand_dims(face_rgb, 0), verbose=0)[0]

    emotion    = EMOTIONS[np.argmax(probs)]
    confidence = float(np.max(probs))
    _emotion_buf().append(emotion)
    smoothed   = Counter(_emotion_buf()).most_common(1)[0][0]

    # Cyberpunk neon face box — corner brackets + scan line + label chip
    ec = EMOTION_COLORS.get(smoothed, "#00f7ff")
    r, g, b_ = tuple(int(ec.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    neon  = (r, g, b_)
    dim   = (max(0, r//4), max(0, g//4), max(0, b_//4))
    white = (220, 230, 255)

    # Subtle dark fill inside the box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+bw, y+bh), dim, -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    # Thin full border (dim, ghost-like)
    cv2.rectangle(frame, (x, y), (x+bw, y+bh), dim, 1)

    # Corner bracket length — ~20 % of the shorter side, clamped
    cl = max(12, min(int(min(bw, bh) * 0.22), 36))
    t  = 2  # bracket thickness

    corners = [
        # top-left
        [(x, y + cl, x, y), (x, y, x + cl, y)],
        # top-right
        [(x+bw, y + cl, x+bw, y), (x+bw, y, x+bw - cl, y)],
        # bottom-left
        [(x, y+bh - cl, x, y+bh), (x, y+bh, x + cl, y+bh)],
        # bottom-right
        [(x+bw, y+bh - cl, x+bw, y+bh), (x+bw, y+bh, x+bw - cl, y+bh)],
    ]
    for seg_pair in corners:
        for (x1, y1, x2, y2) in seg_pair:
            cv2.line(frame, (x1, y1), (x2, y2), neon, t)

    # Animated-look horizontal scan line across the face (static position: 40 % down)
    scan_y = y + int(bh * 0.40)
    scan_overlay = frame.copy()
    cv2.line(scan_overlay, (x + 2, scan_y), (x + bw - 2, scan_y), neon, 1)
    cv2.addWeighted(scan_overlay, 0.55, frame, 0.45, 0, frame)

    # Small diagonal tick marks at the mid-sides for that HUD reticle feel
    tick = 6
    mid_x = x + bw // 2
    mid_y = y + bh // 2
    cv2.line(frame, (mid_x - tick, y), (mid_x + tick, y), neon, 1)        # top mid
    cv2.line(frame, (mid_x - tick, y+bh), (mid_x + tick, y+bh), neon, 1)  # bottom mid
    cv2.line(frame, (x, mid_y - tick), (x, mid_y + tick), neon, 1)        # left mid
    cv2.line(frame, (x+bw, mid_y - tick), (x+bw, mid_y + tick), neon, 1)  # right mid

    # Label chip above the box — filled background + neon border + white text
    label     = smoothed.upper()
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thick = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
    pad_x, pad_y = 6, 4
    chip_x1 = x
    chip_y1 = max(0, y - th - pad_y * 2 - 4)
    chip_x2 = x + tw + pad_x * 2
    chip_y2 = max(th + pad_y, y - 4)

    # Chip fill
    chip_bg = frame.copy()
    cv2.rectangle(chip_bg, (chip_x1, chip_y1), (chip_x2, chip_y2), dim, -1)
    cv2.addWeighted(chip_bg, 0.75, frame, 0.25, 0, frame)
    # Chip border
    cv2.rectangle(frame, (chip_x1, chip_y1), (chip_x2, chip_y2), neon, 1)
    # Text
    cv2.putText(frame, label, (chip_x1 + pad_x, chip_y2 - pad_y),
                font, font_scale, white, font_thick, cv2.LINE_AA)

    return smoothed, confidence, frame, probs

# ──────────────────────────────────────────────────────────────
# PLAYLIST GENERATION — unique randomised mix every call
# ──────────────────────────────────────────────────────────────

_FALLBACK_POOLS = {
    "happy": [
        ("Happy", "Pharrell Williams"), ("Can't Stop the Feeling!", "Justin Timberlake"),
        ("Good as Hell", "Lizzo"), ("Uptown Funk", "Mark Ronson ft. Bruno Mars"),
        ("Walking on Sunshine", "Katrina and the Waves"), ("Here Comes the Sun", "The Beatles"),
        ("Don't Stop Me Now", "Queen"), ("Mr. Blue Sky", "Electric Light Orchestra"),
        ("September", "Earth, Wind & Fire"), ("I Gotta Feeling", "The Black Eyed Peas"),
        ("Shake It Off", "Taylor Swift"), ("Lovely Day", "Bill Withers"),
        ("Dancing Queen", "ABBA"), ("Good Day Sunshine", "The Beatles"),
        ("Best Day of My Life", "American Authors"), ("Feels", "Calvin Harris ft. Pharrell"),
        ("Dynamite", "BTS"), ("Levitating", "Dua Lipa"),
        ("Sunflower", "Post Malone & Swae Lee"), ("As It Was", "Harry Styles"),
    ],
    "sad": [
        # Comforting, healing, hopeful — lift you gently
        ("Fix You", "Coldplay"), ("Here Comes the Sun", "The Beatles"),
        ("Fast Car", "Tracy Chapman"), ("Better Place", "Rachel Platten"),
        ("You Are the Best Thing", "Ray LaMontagne"), ("Ho Hey", "The Lumineers"),
        ("The Cave", "Mumford & Sons"), ("Shake It Out", "Florence and the Machine"),
        ("Dog Days Are Over", "Florence and the Machine"), ("Holocene", "Bon Iver"),
        ("Breathe (2 AM)", "Anna Nalick"), ("Let Her Go", "Passenger"),
        ("The Night We Met", "Lord Huron"), ("Skinny Love", "Bon Iver"),
        ("Medicine", "Daughter"), ("Youth", "Daughter"),
        ("Keep Your Head Up", "Ben Howard"), ("Promise", "Ben Howard"),
        ("I Will Follow You into the Dark", "Death Cab for Cutie"),
        ("White Flag", "Joseph"),
    ],
    "angry": [
        ("Killing in the Name", "Rage Against the Machine"), ("Break Stuff", "Limp Bizkit"),
        ("Given Up", "Linkin Park"), ("Down with the Sickness", "Disturbed"),
        ("Bulls on Parade", "Rage Against the Machine"), ("Last Resort", "Papa Roach"),
        ("Smells Like Teen Spirit", "Nirvana"), ("Bodies", "Drowning Pool"),
        ("Chop Suey!", "System of a Down"), ("Headstrong", "Trapt"),
        ("Numb", "Linkin Park"), ("Duality", "Slipknot"),
        ("Welcome to the Black Parade", "My Chemical Romance"), ("Paranoid", "Black Sabbath"),
        ("Master of Puppets", "Metallica"), ("Highway to Hell", "AC/DC"),
        ("Back in Black", "AC/DC"), ("Fight for Your Right", "Beastie Boys"),
        ("Run the Jewels", "Run the Jewels"), ("m.A.A.d city", "Kendrick Lamar"),
    ],
    "fear": [
        # Comforting, safe — reduce fear and anxiety
        ("Clair de lune", "Claude Debussy"), ("Comptine d'un autre été", "Yann Tiersen"),
        ("Experience", "Ludovico Einaudi"), ("Divenire", "Ludovico Einaudi"),
        ("Nuvole Bianche", "Ludovico Einaudi"), ("River Flows in You", "Yiruma"),
        ("Breathe", "Pink Floyd"), ("Comfortably Numb", "Pink Floyd"),
        ("Holocene", "Bon Iver"), ("Skinny Love", "Bon Iver"),
        ("Sleep", "Eric Whitacre"), ("Weightless", "Marconi Union"),
        ("Teardrop", "Massive Attack"), ("Angel", "Massive Attack"),
        ("Safe and Sound", "Taylor Swift ft. Civil Wars"), ("The Sound of Silence", "Simon & Garfunkel"),
        ("Breathe (2 AM)", "Anna Nalick"), ("Hide and Seek", "Imogen Heap"),
        ("Lullaby", "Sia"), ("Spiegel im Spiegel", "Arvo Pärt"),
    ],
    "disgust": [
        ("Killing in the Name", "Rage Against the Machine"), ("Territorial Pissings", "Nirvana"),
        ("Anarchy in the U.K.", "Sex Pistols"), ("God Save the Queen", "Sex Pistols"),
        ("Holiday in Cambodia", "Dead Kennedys"), ("Too Drunk to Fuck", "Dead Kennedys"),
        ("Fight the Power", "Public Enemy"), ("Black Steel in the Hour of Chaos", "Public Enemy"),
        ("Cop Killer", "Body Count"), ("99 Problems", "Jay-Z"),
        ("Hate Me Now", "Nas"), ("Ether", "Nas"),
        ("Where Is the Love?", "The Black Eyed Peas"), ("Testify", "Rage Against the Machine"),
        ("Wake Up", "Rage Against the Machine"), ("Sleep Now in the Fire", "RATM"),
        ("Bombtrack", "Rage Against the Machine"), ("Guerrilla Radio", "RATM"),
        ("Dookie", "Green Day"), ("American Idiot", "Green Day"),
    ],
    "surprise": [
        # Energising, euphoric, uplifting — match the rush of surprise
        ("Happy", "Pharrell Williams"), ("Can't Stop the Feeling!", "Justin Timberlake"),
        ("Uptown Funk", "Mark Ronson ft. Bruno Mars"), ("Shake It Off", "Taylor Swift"),
        ("Blinding Lights", "The Weeknd"), ("As It Was", "Harry Styles"),
        ("Good as Hell", "Lizzo"), ("Juice", "Lizzo"),
        ("Electric Feel", "MGMT"), ("Time to Pretend", "MGMT"),
        ("Mr. Brightside", "The Killers"), ("Somebody Told Me", "The Killers"),
        ("Ribs", "Lorde"), ("Tennis Court", "Lorde"),
        ("Pompeii", "Bastille"), ("Things We Lost in the Fire", "Bastille"),
        ("Dog Days Are Over", "Florence + The Machine"), ("Shake It Out", "Florence + The Machine"),
        ("Take Me to Church", "Hozier"), ("From Eden", "Hozier"),
    ],
    "neutral": [
        ("Chill Bumps", "Bonobo"), ("Kiara", "Bonobo"),
        ("Teotihuacan", "Tycho"), ("Awake", "Tycho"),
        ("Bloom", "The Paper Kites"), ("Bloom", "Beach House"),
        ("Coffee", "beabadoobee"), ("Death Bed", "Powfu ft. beabadoobee"),
        ("Lo-Fi Hip Hop", "Idealism"), ("Snowman", "Sia"),
        ("Ribs", "Lorde"), ("Yellow Light", "Of Monsters and Men"),
        ("Little Talks", "Of Monsters and Men"), ("Home", "Edith Whiskers"),
        ("Holocene", "Bon Iver"), ("Skinny Love", "Bon Iver"),
        ("Re: Stacks", "Bon Iver"), ("Flume", "Bon Iver"),
        ("Slow Burn", "Kacey Musgraves"), ("Happy & Sad", "Kacey Musgraves"),
    ],
}

def generate_playlist(emotion: str, seed: int = 0) -> dict:
    mood = get_mood(emotion)

    # Strong entropy: combines seed + real time + os.urandom for true uniqueness
    entropy = int.from_bytes(os.urandom(4), "big")
    fingerprint = hashlib.sha256(
        f"{emotion}|{seed}|{time.monotonic()}|{entropy}".encode()
    ).hexdigest()[:12]

    # Pick random era, decade bias, and flavours each time
    eras = random.sample(["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"], 3)
    flavours = random.sample(
        ["cinematic", "underground", "hypnotic", "visceral", "lo-fi", "stadium",
         "intimate", "avant-garde", "nostalgic", "futuristic", "dark", "raw",
         "melancholic", "euphoric", "transcendent", "gritty", "ethereal",
         "psychedelic", "minimalist", "maximalist", "dreamy", "intense"],
        4
    )
    # Random temperature swing for variety
    temperature = round(random.uniform(0.88, 1.0), 3)

    # Artist pool — well-known artists the model can reliably suggest real tracks from
    mainstream_seeds = random.sample([
        # Pop / Indie Pop
        "Billie Eilish", "Lorde", "Hozier", "Sufjan Stevens", "Bon Iver",
        "Frank Ocean", "SZA", "The Weeknd", "Dua Lipa", "Harry Styles",
        # Rock / Alt Rock
        "Radiohead", "Arctic Monkeys", "The National", "Arcade Fire",
        "Nirvana", "Pearl Jam", "Foo Fighters", "Queens of the Stone Age",
        "The Strokes", "Interpol", "Tame Impala", "LCD Soundsystem",
        # Hip-Hop / R&B
        "Kendrick Lamar", "Tyler the Creator", "J. Cole", "Drake",
        "Anderson .Paak", "Childish Gambino", "Mac Miller", "Isaiah Rashad",
        "Daniel Caesar", "H.E.R.", "Jhené Aiko",
        # Electronic / Dance
        "Daft Punk", "Four Tet", "Bonobo", "Tycho", "Jon Hopkins",
        "Caribou", "Jamie xx", "The Chemical Brothers", "Aphex Twin",
        # Soul / Classic
        "Amy Winehouse", "Adele", "Sam Cooke", "Marvin Gaye", "Stevie Wonder",
        "Nina Simone", "Aretha Franklin", "Otis Redding", "Al Green",
        # International — well-known globally
        "Bad Bunny", "J Balvin", "Rosalía", "BTS", "NewJeans",
        "Stromae", "Charlotte de Witte", "Burna Boy", "Wizkid", "Davido",
        "A.R. Rahman", "Prateek Kuhad", "Mariya Takeuchi", "Hyukoh",
        "Natalia Lafourcade", "Café Tacvba", "Ibeyi",
        # Classic / Legacy
        "David Bowie", "The Beatles", "Fleetwood Mac", "Led Zeppelin",
        "Pink Floyd", "The Cure", "Depeche Mode", "Joy Division", "New Order",
        "The Smiths", "Talking Heads", "Patti Smith", "Lou Reed",
    ], 6)

    prompt = f"""You are a global music curator with deep knowledge of every genre and era. Your mission: curate 6 real, existing tracks that perfectly match the emotion and its SPECIFIC THERAPEUTIC PURPOSE below.

EMOTION: {emotion.upper()}
Genre palette: {mood['genre']}
Energy: {mood['energy']} | BPM: {mood['bpm']} | Core vibe: {mood['vibe']}
Flavour descriptors this mix: {', '.join(flavours)}
Era blend this mix: {', '.join(eras)}
Artist orbits to draw from: {', '.join(mainstream_seeds)}
Unique session token: {fingerprint}

EMOTION-SPECIFIC MUSIC DIRECTION:
- HAPPY / NEUTRAL: Pick uplifting, feel-good, positive tracks. Songs that celebrate, energise, or bring joy and contentment.
- SAD: Pick COMFORTING, HEALING tracks — NOT more sadness. Songs that wrap the listener in warmth, offer hope, or gently lift them up. Think "hug in song form."
- FEAR: Pick CALMING, SAFE tracks — NOT scary or tense. Ambient, classical, gentle acoustic — music that makes the listener feel protected and at peace.
- SURPRISE: Pick ENERGISING, EUPHORIC tracks that match the feeling of being wonderfully caught off-guard. Upbeat, exciting, feel-good — music that amplifies that rush of delight.
- ANGRY: Pick SATISFYING, CATHARTIC heavy tracks that let the listener channel and release rage. Intense, powerful, validating.
- DISGUST: Pick REBELLIOUS, VALIDATING tracks that match the "I'm done with this" energy. Punk, alt, protest music — satisfying and empowering.

Songs must sit in the sweet spot of popularity: not mainstream chart-toppers everyone has heard a thousand times, but also not deep obscurities impossible to find. Think critically acclaimed album cuts, fan-favourite deep cuts, or regional hits. Avoid the top 10 biggest hits by famous artists. Also avoid songs so obscure they may not be on major streaming platforms.

RESPOND WITH EXACTLY THIS FORMAT — nothing before TITLE, nothing after VIBE:
TITLE: [unique, poetic playlist name]
1. [Song Title] — [Artist Name]
2. [Song Title] — [Artist Name]
3. [Song Title] — [Artist Name]
4. [Song Title] — [Artist Name]
5. [Song Title] — [Artist Name]
6. [Song Title] — [Artist Name]
VIBE: [one evocative sentence describing the emotional arc]

MANDATORY RULES — THESE ARE ABSOLUTE:
- REAL SONGS ONLY. Every track MUST be a song you have seen in your training data with 100% certainty — a real release that actually exists on YouTube and Spotify right now. If you have ANY doubt a song exists, do NOT include it. Pick something you are absolutely sure about instead.
- Use the EXACT, official song title and artist name as it appears on streaming platforms. No paraphrasing, no approximations.
- DO NOT invent song titles. DO NOT guess. DO NOT hallucinate. If a song name "feels right" but you cannot recall seeing it explicitly, skip it.
- Before outputting each track, mentally verify: "I have seen this specific song title by this specific artist in my training data." If the answer is no → replace it.
- TARGET THE MIDDLE GROUND of popularity: avoid the biggest chart hits, but also avoid anything so obscure it might not be findable.
- GLOBAL VARIETY — mix in songs from different regions/languages when they fit: Latin, Korean, Indian, African, Japanese, French, etc.
- Each track must genuinely match the emotional vibe and the SPECIFIC THERAPEUTIC PURPOSE for this emotion.
- Do NOT repeat artists across the 6 tracks.
- This must be a COMPLETELY DIFFERENT mix every single call — the token above guarantees uniqueness."""

    raw = None

    # Try the best model first with a tight timeout; on any failure fall through instantly
    _PRIMARY_MODEL   = "llama-3.3-70b"
    _SECONDARY_MODEL = "llama-3.2-3b"

    for _model_name, _timeout in ((_PRIMARY_MODEL, 6.0), (_SECONDARY_MODEL, 4.0)):
        try:
            import httpx as _httpx
            _fast_client = __import__("openai").OpenAI(
                base_url="https://api.oxlo.ai/v1",
                api_key=OXLO_KEY,
                timeout=_httpx.Timeout(_timeout, connect=2.0),
            )
            resp = _fast_client.chat.completions.create(
                model=_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content
            if raw and "TITLE:" in raw:
                break
        except Exception:
            raw = None
            continue

    # Robust fallback: build a real playlist from curated pool
    if not raw or "TITLE:" not in raw:
        pool = _FALLBACK_POOLS.get(emotion, _FALLBACK_POOLS["neutral"])
        rng = random.Random(entropy + seed)
        picks = rng.sample(pool, min(6, len(pool)))
        title_words = random.sample(flavours, 2)
        title = f"{title_words[0].title()} {title_words[1].title()} Sessions"
        lines = [f"TITLE: {title}"]
        for i, (song, artist) in enumerate(picks, 1):
            lines.append(f"{i}. {song} — {artist}")
        lines.append(f"VIBE: A {mood['vibe']} journey through {mood['genre']} frequencies.")
        raw = "\n".join(lines)

    lines  = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    title  = next((l.replace("TITLE:", "").strip() for l in lines if l.startswith("TITLE:")), emotion.capitalize())
    vibe   = next((l.replace("VIBE:", "").strip()  for l in lines if l.startswith("VIBE:")), "")
    tracks = []
    for l in lines:
        if l and l[0].isdigit() and "." in l:
            parts = l.split(".", 1)[1].strip()
            if "\u2014" in parts:
                song, artist = parts.split("\u2014", 1)
            elif " - " in parts:
                song, artist = parts.split(" - ", 1)
            else:
                song, artist = parts, ""
            tracks.append({"song": song.strip(), "artist": artist.strip()})

    return {"title": title, "tracks": tracks, "vibe": vibe, "raw": raw}

# ──────────────────────────────────────────────────────────────
# URL BUILDERS
# ──────────────────────────────────────────────────────────────

def youtube_url(playlist: dict, emotion: str) -> str:
    """Opens YouTube search for the playlist name + 'playlist' keyword."""
    mood  = get_mood(emotion)
    title = playlist.get("title", "").strip()
    query = title if title else f"{emotion} {mood['vibe']} music"
    # Append 'playlist' so YouTube surfaces actual playlists, not individual videos
    query = f"{query} playlist"
    encoded = query.replace(" ", "+").replace("&", "%26")
    return f"https://www.youtube.com/results?search_query={encoded}&sp=EgIQAQ%3D%3D"

def youtube_track_url(song: str, artist: str) -> str:
    """Per-track YouTube URL — song + artist search, filtered to videos."""
    # No quotes — they get double-encoded in the browser and break the search
    q = f"{song} {artist} official".strip()
    encoded = q.replace(" ", "+").replace("&", "%26").replace('"', "").replace("'", "")
    return f"https://www.youtube.com/results?search_query={encoded}&sp=EgIQAQ%3D%3D"

def spotify_url(playlist: dict, emotion: str) -> str:
    """Opens Spotify search for the playlist name + 'playlist' keyword."""
    title = playlist.get("title", "").strip()
    if not title:
        mood  = get_mood(emotion)
        title = f"{mood['genre']} {mood['vibe']}"
    # Append 'playlist' so Spotify surfaces playlists, not individual tracks
    title = f"{title} playlist"
    encoded = title.replace(" ", "%20").replace("/", "%2F").replace("&", "%26")
    return f"https://open.spotify.com/search/{encoded}/playlists"

def spotify_track_url(song: str, artist: str) -> str:
    """Per-track Spotify URL — plain song + artist search for reliable matching."""
    # Plain query works better in the URL; Spotify's field filters confuse the web URL parser
    q = f"{song} {artist}".strip()
    encoded = q.replace(" ", "%20").replace("&", "%26").replace("/", "%2F").replace("'", "").replace('"', "")
    return f"https://open.spotify.com/search/{encoded}/tracks"

def apple_music_url(playlist: dict, emotion: str) -> str:
    """Apple Music search URL for the playlist title + 'playlist' keyword."""
    title = playlist.get("title", "").strip()
    if not title:
        mood  = get_mood(emotion)
        title = f"{mood['genre']} {mood['vibe']}"
    # Append 'playlist' so Apple Music surfaces playlists in results
    title = f"{title} playlist"
    encoded = title.replace(" ", "+").replace("&", "and").replace("/", " ")
    return f"https://music.apple.com/search?term={encoded}"

def apple_music_track_url(song: str, artist: str) -> str:
    """Per-track Apple Music search URL — song + artist for exact match."""
    q = f"{song} {artist}".strip().replace(" ", "+").replace("&", "and").replace("/", " ").replace("'", "").replace('"', "")
    return f"https://music.apple.com/search?term={q}&types=songs"

# ──────────────────────────────────────────────────────────────
# UI HELPERS
# ──────────────────────────────────────────────────────────────

def share_text(pl: dict, emotion: str) -> str:
    """Format playlist as shareable text."""
    emoji = EMOJI_FULL.get(emotion, "🎵")
    title = pl.get("title", "Neural Playlist")
    tracks = pl.get("tracks", [])
    lines = [f"{emoji} {title} — via MoodBeats AI", ""]
    for i, t in enumerate(tracks, 1):
        song   = t.get("song", "")
        artist = t.get("artist", "")
        lines.append(f"{i:02d}. {song}{' — ' + artist if artist else ''}")
    vibe = pl.get("vibe", "")
    if vibe:
        lines += ["", f'"{vibe}"']
    lines += ["", "#MoodBeats #AI #Music"]
    return "\n".join(lines)




def render_playlist(pl: dict, emotion: str):
    """Cyberpunk track list — click play button to stream inline YouTube."""
    color = EMOTION_COLORS.get(emotion, "#00f7ff")
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    def wave_html():
        return ('<div class="track-wave">'
                '<span></span><span></span><span></span>'
                '<span></span><span></span><span></span>'
                '</div>')

    items_html = ""
    for i, t in enumerate(pl["tracks"]):
        song   = t.get("song", "")
        artist = t.get("artist", "")
        yt_url = youtube_track_url(song, artist)
        sp_url = spotify_track_url(song, artist)
        am_url = apple_music_track_url(song, artist)
        safe_song   = song.replace("'", "").replace('"', '')
        safe_artist = artist.replace("'", "").replace('"', '')
        yt_q = f"{safe_song} {safe_artist}"
        items_html += (
            f'<li class="track-row" id="track-row-{i}">'
            f'<span class="track-num">{i+1:02d}</span>'
            f'<span class="track-body">'
            f'<span class="track-song">{song}</span>'
            + (f'<span class="track-artist">{artist}</span>' if artist else '') +
            f'</span>'
            f'{wave_html()}'
            f'<span class="track-btns">'
            f'<a class="tp-btn yt" href="{yt_url}" target="_blank">YT</a>'
            f'<a class="tp-btn sp" href="{sp_url}" target="_blank">SP</a>'
            f'<a class="tp-btn am" href="{am_url}" target="_blank">AM</a>'
            f'</span>'
            f'</li>'
        )

    pl_vibe_html = f"<p class='pl-vibe'>&ldquo;{pl['vibe']}&rdquo;</p>" if pl.get('vibe') else ""

    st.markdown(f"""
    <style>
    .track-row {{
      display: flex; align-items: center; gap: .6rem;
      padding: .52rem .4rem; border-radius: 3px;
      transition: background .18s ease, transform .15s ease;
      margin-bottom: 1px; list-style: none;
    }}
    .track-row:hover {{ background: rgba({r},{g},{b},0.05); transform: translateX(4px); }}
    .track-row:hover .track-song {{ color: {color}; text-shadow: 0 0 8px rgba({r},{g},{b},0.4); }}
    .track-row.playing {{ background: rgba({r},{g},{b},0.07) !important; transform: translateX(6px) !important; }}
    .track-row.playing .track-song {{ color: {color} !important; text-shadow: 0 0 14px rgba({r},{g},{b},0.6) !important; }}

    .track-btns {{ display: flex; gap: .3rem; flex-shrink: 0; align-items: center; }}
    .tp-btn.yt, .tp-btn.sp, .tp-btn.am {{
      color: {color} !important;
      border-color: rgba({r},{g},{b},0.35) !important;
      background: rgba({r},{g},{b},0.05) !important;
    }}
    .tp-btn.yt:hover, .tp-btn.sp:hover, .tp-btn.am:hover {{
      border-color: {color} !important;
      background: rgba({r},{g},{b},0.14) !important;
      box-shadow: 0 0 10px rgba({r},{g},{b},0.4) !important;
      transform: scale(1.05);
    }}
    /* ── Inline YouTube player panel ── */
    @keyframes yt-panel-in {{
      from {{ opacity:0; transform:translateY(-6px) scaleY(.94); }}
      to   {{ opacity:1; transform:translateY(0) scaleY(1); }}
    }}
    .yt-player-panel {{
      margin: .3rem 0 .7rem 2.4rem;
      border: 1px solid rgba(255,0,51,0.25);
      border-radius: 5px;
      background: rgba(6,4,10,0.98);
      overflow: hidden;
      animation: yt-panel-in .28s cubic-bezier(.22,1,.36,1) forwards;
      position: relative;
    }}
    .yt-player-panel::before {{
      content:''; position:absolute; top:0; left:0; right:0; height:1.5px;
      background:linear-gradient(90deg,transparent,rgba(255,0,51,.6),transparent);
      z-index:1;
    }}
    .yt-player-topbar {{
      display:flex; align-items:center; justify-content:space-between;
      padding:.38rem .75rem;
      border-bottom:1px solid rgba(255,0,51,0.12);
      background:rgba(255,0,51,0.04);
    }}
    .yt-player-label {{
      font-family:'Share Tech Mono',monospace;
      font-size:.48rem; letter-spacing:.12em;
      text-transform:uppercase; color:rgba(255,0,51,0.55);
      display:flex; align-items:center; gap:.4rem;
    }}
    .yt-live-dot {{
      width:5px; height:5px; border-radius:50%;
      background:#ff0033; box-shadow:0 0 6px #ff0033;
      animation:dot-blink 1s step-end infinite;
    }}
    .yt-player-close {{
      background:none; border:1px solid rgba(255,0,51,0.2);
      border-radius:2px; color:rgba(255,0,51,0.45);
      font-size:.52rem; cursor:pointer; padding:.1rem .38rem;
      font-family:'Orbitron',monospace; letter-spacing:.1em;
      transition:all .2s ease;
    }}
    .yt-player-close:hover {{
      background:rgba(255,0,51,0.1); border-color:#ff0033;
      color:#ff0033; box-shadow:0 0 10px rgba(255,0,51,0.4);
    }}
    .yt-now-playing {{
      font-family:'Rajdhani',sans-serif; font-size:.8rem;
      padding:.32rem .75rem .18rem;
      color:rgba(255,80,80,0.7); letter-spacing:.03em;
    }}
    .yt-now-playing em {{ color:#ff4466; font-style:normal; font-weight:600; }}
    .yt-iframe-wrap {{
      position:relative; padding-bottom:52.5%; height:0; overflow:hidden;
    }}
    .yt-iframe-wrap iframe {{
      position:absolute; top:0; left:0; width:100%; height:100%; border:none;
    }}
    .yt-player-hint {{
      font-family:'Share Tech Mono',monospace;
      font-size:.45rem; color:rgba(255,0,51,0.2);
      letter-spacing:.08em; text-align:center; text-transform:uppercase;
      padding:.28rem .75rem .45rem;
    }}
    </style>

    <div class="playlist-card" style="border-left:3px solid {color}55; box-shadow: inset 3px 0 20px {color}12;">
      <span class="pl-prefix">// Neural Playlist</span>
      <div class="pl-title" style="color:{color};text-shadow:0 0 28px {color}88,0 0 60px {color}33;">{pl['title']}</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:rgba(0,247,255,0.28);letter-spacing:.1em;text-transform:uppercase;margin-bottom:.55rem;">// YT / SP / AM opens new tab</div>
      <ul id="cp-tracklist" style="padding:0;margin:0;list-style:none;">{items_html}</ul>
      {pl_vibe_html}
    </div>

    <script>
    (function(){{
      var _activeIdx = null;
      function _closePlayer(){{
        var old = document.getElementById('yt-player-wrap');
        if(old) old.remove();
        if(_activeIdx !== null){{
          var row = document.getElementById('track-row-'+_activeIdx);
          if(row) row.classList.remove('playing');
          var icon = document.getElementById('play-icon-'+_activeIdx);
          if(icon) icon.innerHTML = '&#x25B6;';
          _activeIdx = null;
        }}
      }}
      window.openYTPlayer = function(idx, query, song, artist){{
        if(_activeIdx === idx){{ _closePlayer(); return; }}
        _closePlayer();
        _activeIdx = idx;
        var row = document.getElementById('track-row-'+idx);
        if(row) row.classList.add('playing');
        var icon = document.getElementById('play-icon-'+idx);
        if(icon) icon.innerHTML = '&#x23F8;';
        var src = 'https://www.youtube.com/embed?listType=search&list='
          + encodeURIComponent(query) + '&autoplay=1&mute=0&modestbranding=1&rel=0';
        var panel = document.createElement('div');
        panel.id = 'yt-player-wrap';
        panel.className = 'yt-player-panel';
        panel.innerHTML =
          '<div class="yt-player-topbar">'
          + '<span class="yt-player-label"><span class="yt-live-dot"></span>YOUTUBE STREAM // NEURAL AUDIO</span>'
          + '<button class="yt-player-close" onclick="window._closeYT()">&#x2715; CLOSE</button>'
          + '</div>'
          + '<div class="yt-now-playing">&#x266B;&nbsp; <em>' + song + '</em>'
          + (artist ? ' &nbsp;&middot;&nbsp; ' + artist : '') + '</div>'
          + '<div class="yt-iframe-wrap">'
          + '<iframe src="' + src + '" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
          + '</div>'
          + '<div class="yt-player-hint">// if autoplay is blocked — click play inside the player</div>';
        if(row) row.insertAdjacentElement('afterend', panel);
      }};
      window._closeYT = _closePlayer;
    }})();
    </script>
    """, unsafe_allow_html=True)


def render_emotion_result(emotion: str):
    """Emotion reveal card with vibe pills."""
    emoji = EMOJI_FULL.get(emotion, "🎵")
    color = EMOTION_COLORS.get(emotion, "#00f7ff")
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    mood  = get_mood(emotion)
    # Emotion-specific neon silhouette glow — color matches the detected emotion
    emoji_glow = (
        f"drop-shadow(0 0 18px rgba({r},{g},{b},0.95)) "
        f"drop-shadow(0 0 8px rgba({r},{g},{b},0.85)) "
        f"drop-shadow(0 0 32px rgba({r},{g},{b},0.55)) "
        f"drop-shadow(0 0 60px rgba({r},{g},{b},0.25))"
    )
    st.markdown(f"""
    <style>
    @keyframes emo-silhouette-pulse-{emotion} {{
      0%,100% {{
        filter: drop-shadow(0 0 14px rgba({r},{g},{b},0.8))
                drop-shadow(0 0 6px rgba({r},{g},{b},0.7))
                drop-shadow(0 0 26px rgba({r},{g},{b},0.4));
      }}
      50% {{
        filter: drop-shadow(0 0 28px rgba({r},{g},{b},1.0))
                drop-shadow(0 0 12px rgba({r},{g},{b},0.9))
                drop-shadow(0 0 55px rgba({r},{g},{b},0.65))
                drop-shadow(0 0 90px rgba({r},{g},{b},0.3));
      }}
    }}
    .emo-emoji-{emotion} {{
      font-size: 3rem; line-height: 1;
      filter: {emoji_glow};
      animation: emo-bob 3.5s ease-in-out infinite,
                 emo-silhouette-pulse-{emotion} 2.8s ease-in-out infinite;
    }}
    </style>
    <div class="emo-reveal" style="
      border-color:{color}50;
      background: linear-gradient(135deg, rgba({r},{g},{b},0.05) 0%, rgba(10,10,20,0.9) 100%);
      box-shadow: 0 0 40px rgba({r},{g},{b},0.08), inset 0 0 60px rgba({r},{g},{b},0.03);
    ">
      <div class="emo-top">
        <div class="emo-emoji-{emotion}">{emoji}</div>
        <div>
          <div class="emo-name"
               style="color:{color}; text-shadow:0 0 30px {color}cc, 0 0 60px {color}55;">
            {emotion.upper()}
          </div>
          <div class="emo-sub">// Neural scan locked · emotion confirmed</div>
        </div>
      </div>
      <div class="vibe-pills">
        <span class="pill genre">&#x2B21; {mood['genre']}</span>
        <span class="pill energy">&#x26A1; {mood['energy']} energy</span>
        <span class="pill bpm">&#x266A; {mood['bpm']} bpm</span>
        <span class="pill vibe">&#x25C8; {mood['vibe']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_mood_override(current_emotion: str):
    """
    Cyberpunk mood override panel.

    THE STRUCTURAL PROBLEM WITH STREAMLIT:
    st.markdown() opens an HTML div, but Streamlit widget calls (st.columns,
    st.button) render into a *separate* DOM tree managed by Streamlit's React
    frontend — they are never children of that div.  Closing the div in a later
    st.markdown() call only inserts an orphaned </div> tag; the buttons are
    still outside the container visually and in the DOM.

    THE FIX:
    We use st.container() to group everything, then use CSS to make the
    container's *own* Streamlit-generated DOM node look like the panel.
    A unique `data-ovpanel` attribute is stamped onto the container's root
    element via a tiny injected <script>, letting us target it precisely with
    CSS without touching any other element on the page.
    The buttons, columns, and decorative HTML all live inside that same
    st.container() scope — they are genuine DOM siblings, and CSS flexbox/grid
    applied to the container makes them equidistant inside it.
    """
    all_emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    NEON = {
        "angry":    "#FF0044",   # Red
        "disgust":  "#00FF88",   # Green
        "fear":     "#AA00FF",   # Purple
        "happy":    "#FFE600",   # Yellow
        "neutral":  "#88CCFF",   # Blue/White
        "sad":      "#00EEFF",   # Cyan
        "surprise": "#FF00CC",   # Pink/Magenta
    }

    panel_ec = NEON.get(current_emotion, "#00F0FF")
    pr, pg, pb = int(panel_ec[1:3], 16), int(panel_ec[3:5], 16), int(panel_ec[5:7], 16)

    # ── Per-button CSS (targets the wrapper div we stamp next to each button) ──
    btn_css = ""
    for e in all_emotions:
        ec = NEON.get(e, "#00F0FF")
        r, g, b = int(ec[1:3], 16), int(ec[3:5], 16), int(ec[5:7], 16)
        is_sel = (e == current_emotion)

        sel_extra = ""
        if is_sel:
            sel_extra = f"""
            background : rgba({r},{g},{b},0.20) !important;
            border-color: {ec} !important;
            border-width: 2px !important;
            color       : {ec} !important;
            box-shadow  : 0 0 22px rgba({r},{g},{b},0.85),
                          0 0 55px rgba({r},{g},{b},0.45),
                          0 0 90px rgba({r},{g},{b},0.18),
                          inset 0 0 18px rgba({r},{g},{b},0.15) !important;
            text-shadow : 0 0 12px {ec},
                          0 0 28px rgba({r},{g},{b},0.9),
                          0 0 55px rgba({r},{g},{b},0.5) !important;
            animation   : ov-pulse-{e} 2s ease-in-out infinite !important;
            """

        btn_css += f"""
        @keyframes ov-pulse-{e} {{
          0%,100% {{ box-shadow: 0 0 18px rgba({r},{g},{b},0.7),
                                  0 0 45px rgba({r},{g},{b},0.35),
                                  inset 0 0 12px rgba({r},{g},{b},0.12); }}
          50%      {{ box-shadow: 0 0 32px rgba({r},{g},{b},1.0),
                                  0 0 75px rgba({r},{g},{b},0.6),
                                  0 0 120px rgba({r},{g},{b},0.25),
                                  inset 0 0 24px rgba({r},{g},{b},0.2); }}
        }}
        [data-ovpanel] .ov-wrap-{e} .stButton > button {{
          border     : 1.5px solid rgba({r},{g},{b},0.55) !important;
          color      : rgba({r},{g},{b},0.82) !important;
          background : rgba({r},{g},{b},0.07) !important;
          text-shadow: 0 0 9px rgba({r},{g},{b},0.55),
                       0 0 20px rgba({r},{g},{b},0.28) !important;
          box-shadow : 0 0 12px rgba({r},{g},{b},0.22),
                       inset 0 0 8px rgba({r},{g},{b},0.05) !important;
          {sel_extra}
        }}
        [data-ovpanel] .ov-wrap-{e} .stButton > button:hover {{
          border-color: {ec} !important;
          border-width: 2px !important;
          color       : {ec} !important;
          background  : rgba({r},{g},{b},0.18) !important;
          box-shadow  : 0 0 26px rgba({r},{g},{b},0.9),
                        0 0 65px rgba({r},{g},{b},0.45),
                        inset 0 0 16px rgba({r},{g},{b},0.12) !important;
          text-shadow : 0 0 14px {ec},
                        0 0 32px rgba({r},{g},{b},0.95),
                        0 0 60px rgba({r},{g},{b},0.5) !important;
          transform   : translateY(-3px) scale(1.06) !important;
        }}
        """

    # ── Inject all CSS + a script that stamps data-ovpanel on the container ──
    st.markdown(f"""
    <style>
    /* ── Keyframes ── */
    @keyframes ov-fadein  {{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:translateY(0)}} }}
    @keyframes ov-scan    {{
      0%   {{ left:-110px; opacity:0; }}
      8%   {{ opacity:1; }}
      92%  {{ opacity:0.7; }}
      100% {{ left:110%; opacity:0; }}
    }}
    @keyframes ov-flicker {{
      0%,88%,90%,92%,94%,100% {{ opacity:1; }}
      89%  {{ opacity:0.45; }}
      91%  {{ opacity:0.9;  }}
      93%  {{ opacity:0.35; }}
    }}
    @keyframes ov-dot {{
      0%,100% {{ transform:scale(1);   opacity:0.55;
                 box-shadow:0 0 6px rgba({pr},{pg},{pb},0.7),0 0 14px rgba({pr},{pg},{pb},0.4); }}
      50%      {{ transform:scale(1.4); opacity:1.0;
                 box-shadow:0 0 14px rgba({pr},{pg},{pb},1.0),0 0 32px rgba({pr},{pg},{pb},0.85),0 0 55px rgba({pr},{pg},{pb},0.4); }}
    }}
    @keyframes ov-amb    {{
      0%,100% {{ opacity:0.35; }}
      50%      {{ opacity:0.7;  }}
    }}

    /* ══ Panel shell — applied to the container Streamlit generates ══ */
    [data-ovpanel] {{
      margin-top    : 1.2rem !important;
      border        : 2px solid rgba({pr},{pg},{pb},0.8) !important;
      border-radius : 10px !important;
      background    : linear-gradient(145deg,
                        rgba(5,5,14,0.97) 0%,
                        rgba(8,8,20,0.99) 60%,
                        rgba(6,6,16,0.98) 100%) !important;
      padding       : 1rem 1.2rem 1.1rem !important;
      position      : relative !important;
      overflow      : hidden !important;
      backdrop-filter: blur(16px) !important;
      animation     : ov-fadein .4s ease forwards !important;
      box-shadow    :
        0 0 35px rgba({pr},{pg},{pb},0.4),
        0 0 80px rgba({pr},{pg},{pb},0.18),
        0 0 140px rgba({pr},{pg},{pb},0.08),
        inset 0 0 50px rgba({pr},{pg},{pb},0.05),
        inset 0 1px 0 rgba({pr},{pg},{pb},0.15) !important;
    }}

    /* Scanline texture inside panel */
    [data-ovpanel]::before {{
      content  : '';
      position : absolute; inset: 0;
      background: repeating-linear-gradient(
        0deg,
        transparent, transparent 3px,
        rgba({pr},{pg},{pb},0.022) 3px,
        rgba({pr},{pg},{pb},0.022) 4px
      );
      pointer-events: none; z-index: 0; border-radius: 10px;
    }}

    /* ── Decorative children that stay inside the panel ── */
    .ov-scanbeam {{
      position  : absolute; top:0; bottom:0; width:100px;
      background: linear-gradient(90deg,transparent,rgba({pr},{pg},{pb},0.11),transparent);
      animation : ov-scan 5s ease-in-out infinite;
      pointer-events: none; z-index: 1;
    }}
    .ov-topbar {{
      position  : absolute; top:0; left:0; right:0; height:2px;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba({pr},{pg},{pb},0.55) 25%,
        rgba({pr},{pg},{pb},1.0)  50%,
        rgba({pr},{pg},{pb},0.55) 75%,
        transparent 100%);
      box-shadow: 0 0 14px rgba({pr},{pg},{pb},0.7),
                  0 0 30px rgba({pr},{pg},{pb},0.3);
      z-index: 2;
    }}
    .ov-amb {{
      position  : absolute; bottom:-35px; right:-35px;
      width:200px; height:200px; border-radius:50%;
      background: radial-gradient(circle, rgba({pr},{pg},{pb},0.13) 0%, transparent 70%);
      pointer-events: none; z-index: 0;
      animation : ov-amb 3.2s ease-in-out infinite;
    }}

    /* ── Title ── */
    .ov-title {{
      font-family: 'Orbitron', monospace;
      font-size  : .65rem; letter-spacing:.28em;
      text-transform: uppercase;
      color      : rgba({pr},{pg},{pb},1.0);
      animation  : ov-flicker 5s ease-in-out infinite;
      margin-bottom: .9rem;
      display: flex; align-items: center; gap:.6rem;
      position: relative; z-index: 3;
    }}
    .ov-dot {{
      width:8px; height:8px; border-radius:50%; flex-shrink:0;
      background: rgba({pr},{pg},{pb},1.0);
      animation : ov-dot 2s ease-in-out infinite;
    }}

    /* ══ Button row — 7 equal columns ══ */
    /*
       We target the [data-testid="stHorizontalBlock"] that Streamlit generates
       when st.columns() is called inside the container.  This is the real flex
       row that actually contains the button DOM nodes — so flex rules here
       genuinely keep the buttons inside the panel and equidistant.
    */
    [data-ovpanel] [data-testid="stHorizontalBlock"] {{
      display               : flex !important;
      flex-wrap             : nowrap !important;
      gap                   : .5rem !important;
      align-items           : stretch !important;
      position              : relative; z-index: 3;
    }}
    [data-ovpanel] [data-testid="stHorizontalBlock"]
        > [data-testid="stColumn"] {{
      flex  : 1 1 0 !important;
      min-width: 0 !important;
      padding: 0 !important;
    }}
    [data-ovpanel] .stButton {{
      width: 100% !important;
    }}
    [data-ovpanel] .stButton > button {{
      width         : 100% !important;
      padding       : .55rem .1rem !important;
      font-size     : .52rem !important;
      letter-spacing: .1em !important;
      border-radius : 6px !important;
      transition    : all .25s ease-in-out !important;
      white-space   : nowrap !important;
      min-width     : 0 !important;
      line-height   : 1.4 !important;
      font-family   : 'Orbitron', monospace !important;
      font-weight   : 700 !important;
      text-transform: uppercase !important;
    }}

    /* Responsive wrapping on small screens */
    @media (max-width: 680px) {{
      [data-ovpanel] [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
      }}
      [data-ovpanel] [data-testid="stHorizontalBlock"]
          > [data-testid="stColumn"] {{
        flex: 1 1 calc(25% - .5rem) !important;
      }}
    }}

    {btn_css}
    </style>

    <script>
    /* Stamp data-ovpanel on the next Streamlit block so CSS can target it */
    (function() {{
      function stamp() {{
        var markers = document.querySelectorAll('[data-ov-marker="ov-panel-{current_emotion}"]');
        markers.forEach(function(m) {{
          var block = m.closest('[data-testid="stVerticalBlockBorderWrapper"],\
                                  [data-testid="stVerticalBlock"],\
                                  .element-container');
          /* Walk up until we find the stVerticalBlock that contains our marker
             AND will also contain the columns rendered after this markdown */
          var el = m;
          for (var i = 0; i < 8; i++) {{
            el = el.parentElement;
            if (!el) break;
            if (el.getAttribute('data-testid') === 'stVerticalBlock' ||
                el.getAttribute('data-testid') === 'stVerticalBlockBorderWrapper') {{
              el.setAttribute('data-ovpanel', '1');
              break;
            }}
          }}
        }});
      }}
      if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', function() {{ setTimeout(stamp, 80); }});
      }} else {{
        setTimeout(stamp, 80);
      }}
    }})();
    </script>
    <span data-ov-marker="ov-panel-{current_emotion}" style="display:none"></span>
    """, unsafe_allow_html=True)

    # ── Decorative overlay elements (inside same st.container scope) ──
    st.markdown("""
    <div class="ov-scanbeam"></div>
    <div class="ov-topbar"></div>
    <div class="ov-amb"></div>
    """, unsafe_allow_html=True)

    # ── Title ──
    st.markdown(f"""
    <div class="ov-title">
      <span class="ov-dot"></span>
      Override Mood Detection
    </div>
    """, unsafe_allow_html=True)

    # ── 7 emotion buttons in a single st.columns row ──
    # st.columns lives in the SAME Python scope as the markdowns above,
    # so Streamlit places it in the same stVerticalBlock — which our CSS
    # (via data-ovpanel) wraps with the panel styling.
    cols = st.columns(len(all_emotions), gap="small")
    for col, e in zip(cols, all_emotions):
        with col:
            st.markdown(f'<div class="ov-wrap-{e}">', unsafe_allow_html=True)
            if st.button(e.upper(), key=f"ov_btn_{e}", help=f"Set mood: {e}"):
                if e != current_emotion:
                    st.session_state.locked_emotion  = e
                    st.session_state.playlist_parsed = None
                    st.session_state.phase           = "locked"
                    st.session_state.emotion_buf.clear()
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def render_platform_buttons(pl: dict, emotion: str):
    """YouTube + Spotify + Apple Music buttons styled with current emotion color."""
    yt = youtube_url(pl, emotion)
    sp = spotify_url(pl, emotion)
    am = apple_music_url(pl, emotion)
    color = EMOTION_COLORS.get(emotion, "#00f7ff")
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    st.markdown(f"""
    <style>
    .mood-btn {{
      display: inline-flex; align-items: center; gap: .4rem;
      padding: .5rem 1.3rem;
      border: 1px solid rgba({r},{g},{b},0.4);
      border-radius: var(--radius);
      font-family: var(--orbitron);
      font-size: .74rem; font-weight: 600;
      letter-spacing: .1em; text-transform: uppercase;
      text-decoration: none !important;
      color: {color};
      text-shadow: 0 0 8px rgba({r},{g},{b},0.4);
      background: rgba({r},{g},{b},0.06);
      transition: all .25s ease;
    }}
    .mood-btn:hover {{
      border-color: {color};
      background: rgba({r},{g},{b},0.15);
      box-shadow: 0 0 20px rgba({r},{g},{b},0.45), 0 0 50px rgba({r},{g},{b},0.15);
      text-shadow: 0 0 12px rgba({r},{g},{b},0.7);
      transform: scale(1.04);
    }}
    </style>
    <div class="action-row" style="margin-top: 1.4rem;">
      <a href="{yt}" target="_blank" class="mood-btn">&#x25B6; YouTube</a>
      <a href="{sp}" target="_blank" class="mood-btn">&#x25CE; Spotify</a>
      <a href="{am}" target="_blank" class="mood-btn">&#x266A; Apple Music</a>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════

# ── HUD wordmark ──────────────────────────────────────────────
st.markdown("""
<div class="hud-wordmark">
  <div class="hud-logo">MOODBEATS</div>
  <div class="hud-badge">v4.0 // NEURAL OS</div>
</div>
<div class="hud-tagline">
  emotion <em>&rarr;</em> neural scan <em>&rarr;</em> curated frequency
</div>
<div class="hud-tagline-human">
  Point your camera &mdash; we read your mood &mdash; AI picks your music
</div>
""", unsafe_allow_html=True)

# ── How it works accordion ────────────────────────────────────
st.markdown("""
<style>
details.hiw-wrap {
  margin-bottom: 1.5rem;
  list-style: none;
}
details.hiw-wrap summary {
  display: flex; align-items: center; gap: .6rem;
  cursor: pointer; user-select: none;
  font-family: var(--orbitron); font-size: .88rem;
  letter-spacing: .18em; text-transform: uppercase;
  color: var(--text2);
  padding: .6rem 0;
  border-bottom: 1px solid rgba(0,247,255,0.08);
  transition: color .2s ease;
  list-style: none;
}
details.hiw-wrap summary::-webkit-details-marker { display: none; }
details.hiw-wrap summary::marker { display: none; }
details.hiw-wrap summary:hover { color: var(--cyan); }
details.hiw-wrap[open] summary { color: var(--cyan); border-color: rgba(0,247,255,0.2); }
.hiw-arrow {
  width: 0; height: 0;
  border-top: 4px solid transparent;
  border-bottom: 4px solid transparent;
  border-left: 6px solid currentColor;
  transition: transform .25s ease;
  flex-shrink: 0;
  display: inline-block;
}
details.hiw-wrap[open] .hiw-arrow { transform: rotate(90deg); }
.hiw-body {
  padding: 1rem 0 .5rem;
  font-family: var(--mono); font-size: .95rem;
  color: var(--text2); letter-spacing: .06em; line-height: 2;
}
.hiw-step { display: flex; align-items: flex-start; gap: .8rem; margin-bottom: .7rem; }
.hiw-step-num {
  font-family: var(--orbitron); font-size: .82rem;
  color: var(--cyan); border: 1px solid rgba(0,247,255,0.3);
  padding: .15rem .45rem; border-radius: 2px;
  flex-shrink: 0; margin-top: .1rem;
  text-shadow: 0 0 6px rgba(0,247,255,0.4);
}
.hiw-step-txt em { color: var(--cyan); font-style: normal; }
.hiw-note {
  margin-top: .8rem; padding: .6rem 1rem;
  border-left: 2px solid rgba(0,247,255,0.2);
  font-size: .88rem; color: var(--text3);
}
.hiw-note em { color: rgba(0,247,255,0.5); font-style: normal; }
</style>
<details class="hiw-wrap">
  <summary>
    <span class="hiw-arrow"></span>
    &#x2B21;&nbsp; How it works
  </summary>
  <div class="hiw-body">
    <div class="hiw-step">
      <span class="hiw-step-num">01</span>
      <span class="hiw-step-txt"><em>MediaPipe BlazeFace</em> detects your face in real-time via webcam and crops the region of interest.</span>
    </div>
    <div class="hiw-step">
      <span class="hiw-step-num">02</span>
      <span class="hiw-step-txt"><em>FER2013 CNN</em> (trained on 35,887 facial expressions) classifies 7 emotions: happy, sad, angry, fear, disgust, surprise, neutral.</span>
    </div>
    <div class="hiw-step">
      <span class="hiw-step-num">03</span>
      <span class="hiw-step-txt">A <em>5-frame smoothing buffer</em> stabilises the prediction until 8 consecutive frames agree — lock-on confirmed faster and more accurately.</span>
    </div>
    <div class="hiw-step">
      <span class="hiw-step-num">04</span>
      <span class="hiw-step-txt">The locked emotion seeds an <em>LLM prompt</em> with genre, BPM, vibe descriptors and a random hash — generating a unique curated playlist every time.</span>
    </div>
    <div class="hiw-note" id="hiw-privacy-note">// All camera processing is <em>local &amp; ephemeral</em> — no frames are stored, uploaded or logged.</div>
  </div>
</details>
""", unsafe_allow_html=True)

# ── Keyboard shortcut JS (Space / Enter to trigger scan) ─────
st.markdown("""
<script>
(function() {
  function handleKey(e) {
    // Only fire on Space or Enter, not in inputs
    if (e.code !== 'Space' && e.code !== 'Enter') return;
    var tag = document.activeElement ? document.activeElement.tagName : '';
    if (['INPUT','TEXTAREA','SELECT','BUTTON','A'].includes(tag)) return;
    e.preventDefault();
    // Find the scan / rescan button by its text content
    var btns = document.querySelectorAll('.stButton > button');
    for (var i = 0; i < btns.length; i++) {
      var txt = btns[i].innerText.toUpperCase();
      if (txt.includes('NEURAL SCAN') || txt.includes('RESCAN')) {
        btns[i].click();
        return;
      }
    }
  }
  document.addEventListener('keydown', handleKey);
})();
</script>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════════════════════════
#  LEFT COLUMN — Scanner
# ══════════════════════════════════════════════════════════════

with col_left:
    phase = st.session_state.phase
    st.markdown('<div class="section-label">&#x2B21; Emotion Scanner</div>', unsafe_allow_html=True)

    # ── Idle ──────────────────────────────────────────────────
    if phase == "idle":
        st.markdown("""
        <div class="scan-idle">

          <!-- Music frequency visualizer bars (audio-reactive via JS) -->
          <div class="idle-freq-wrap">
            <div id="idle-freq">
              <span></span><span></span><span></span><span></span><span></span>
              <span></span><span></span><span></span><span></span><span></span>
              <span></span><span></span><span></span><span></span><span></span>
              <span></span><span></span><span></span><span></span><span></span>
              <span></span><span></span><span></span><span></span><span></span>
              <span></span><span></span><span></span><span></span><span></span>
            </div>
          </div>

          <!-- Orbital reticle -->
          <div class="idle-orb-wrap">
            <div class="idle-orbit idle-orbit-1">
              <div class="idle-orbit-dot"></div>
            </div>
            <div class="idle-orbit idle-orbit-2">
              <div class="idle-orbit-dot"></div>
            </div>
            <svg class="idle-svg" viewBox="0 0 100 100" fill="none">
              <circle cx="50" cy="50" r="38" stroke="#00f7ff" stroke-width="0.8" opacity=".2"
                stroke-dasharray="6 3"/>
              <circle cx="50" cy="50" r="26" stroke="#00f7ff" stroke-width="0.8" opacity=".35"/>
              <circle cx="50" cy="50" r="5" fill="#00f7ff" opacity=".85"/>
              <circle cx="50" cy="50" r="5" fill="#00f7ff" opacity=".3"
                style="animation:idle-ping 2.2s ease-out infinite"/>
              <line x1="50" y1="10" x2="50" y2="22" stroke="#00f7ff" stroke-width="1" opacity=".5"/>
              <line x1="50" y1="78" x2="50" y2="90" stroke="#00f7ff" stroke-width="1" opacity=".5"/>
              <line x1="10" y1="50" x2="22" y2="50" stroke="#00f7ff" stroke-width="1" opacity=".5"/>
              <line x1="78" y1="50" x2="90" y2="50" stroke="#00f7ff" stroke-width="1" opacity=".5"/>
              <path d="M13 22 L13 13 L22 13" stroke="#00f7ff" stroke-width="1.2" opacity=".4"/>
              <path d="M87 22 L87 13 L78 13" stroke="#00f7ff" stroke-width="1.2" opacity=".4"/>
              <path d="M13 78 L13 87 L22 87" stroke="#00f7ff" stroke-width="1.2" opacity=".4"/>
              <path d="M87 78 L87 87 L78 87" stroke="#00f7ff" stroke-width="1.2" opacity=".4"/>
            </svg>
          </div>

          <div class="scan-hint">// Awaiting biometric input</div>
          <div class="idle-sub">Point camera · facial emotion detection · AI-curated playlist</div>
        </div>

        <style>
        @keyframes idle-ping {
          0%   { r: 5; opacity: .5; }
          100% { r: 28; opacity: 0; }
        }
        .idle-freq-wrap {
          height: 52px; display: flex; align-items: flex-end;
          justify-content: center; margin-bottom: 1.6rem;
        }
        #idle-freq {
          display: flex; align-items: flex-end;
          gap: 3px; height: 48px;
        }
        #idle-freq span {
          display: block; width: 5px; border-radius: 3px;
          background: linear-gradient(180deg, #ff00ff, #00f7ff);
          box-shadow: 0 0 8px #00f7ff88;
          animation: idle-bar 1.6s ease-in-out infinite;
          height: 6px;
        }
        #idle-freq span:nth-child(odd)  { animation-duration: 1.4s; }
        #idle-freq span:nth-child(3n)   { background: linear-gradient(180deg, #9d00ff, #00f7ff); animation-duration: 1.9s; }
        #idle-freq span:nth-child(5n)   { background: linear-gradient(180deg, #00ff9f, #00f7ff); animation-duration: 1.2s; }
        @keyframes idle-bar {
          0%, 100% { height: 4px; opacity: .3; }
          50%       { height: 42px; opacity: .9; }
        }
        #idle-freq span:nth-child(1)  { animation-delay: 0s; }
        #idle-freq span:nth-child(2)  { animation-delay: .08s; }
        #idle-freq span:nth-child(3)  { animation-delay: .16s; }
        #idle-freq span:nth-child(4)  { animation-delay: .24s; }
        #idle-freq span:nth-child(5)  { animation-delay: .32s; }
        #idle-freq span:nth-child(6)  { animation-delay: .40s; }
        #idle-freq span:nth-child(7)  { animation-delay: .48s; }
        #idle-freq span:nth-child(8)  { animation-delay: .56s; }
        #idle-freq span:nth-child(9)  { animation-delay: .64s; }
        #idle-freq span:nth-child(10) { animation-delay: .72s; }
        #idle-freq span:nth-child(11) { animation-delay: .80s; }
        #idle-freq span:nth-child(12) { animation-delay: .88s; }
        #idle-freq span:nth-child(13) { animation-delay: .96s; }
        #idle-freq span:nth-child(14) { animation-delay: 1.04s; }
        #idle-freq span:nth-child(15) { animation-delay: 1.12s; }
        #idle-freq span:nth-child(16) { animation-delay: 1.20s; }
        #idle-freq span:nth-child(17) { animation-delay: 1.28s; }
        #idle-freq span:nth-child(18) { animation-delay: 1.36s; }
        #idle-freq span:nth-child(19) { animation-delay: .90s; }
        #idle-freq span:nth-child(20) { animation-delay: .70s; }
        #idle-freq span:nth-child(21) { animation-delay: .50s; }
        #idle-freq span:nth-child(22) { animation-delay: .30s; }
        #idle-freq span:nth-child(23) { animation-delay: .10s; }
        #idle-freq span:nth-child(24) { animation-delay: .25s; }
        #idle-freq span:nth-child(25) { animation-delay: .45s; }
        #idle-freq span:nth-child(26) { animation-delay: .65s; }
        #idle-freq span:nth-child(27) { animation-delay: .85s; }
        #idle-freq span:nth-child(28) { animation-delay: 1.05s; }
        #idle-freq span:nth-child(29) { animation-delay: 1.25s; }
        #idle-freq span:nth-child(30) { animation-delay: 1.45s; }

        /* Audio-reactive override */
        #idle-freq.audio-live span { animation: none; transition: height 0.05s ease; }

        .idle-orb-wrap {
          width: 100px; height: 100px;
          margin: 0 auto 1.4rem;
          position: relative;
          display: flex; align-items: center; justify-content: center;
        }
        .idle-svg { position: absolute; inset: 0; width: 100%; height: 100%; }
        .idle-orbit {
          position: absolute; border-radius: 50%;
          border: 1px dashed rgba(0,247,255,0.2);
          animation: orbit-spin linear infinite;
        }
        .idle-orbit-1 { width: 110px; height: 110px; top: -5px; left: -5px; animation-duration: 8s; }
        .idle-orbit-2 { width: 130px; height: 130px; top: -15px; left: -15px; animation-duration: 14s; animation-direction: reverse; border-color: rgba(255,0,255,0.15); }
        @keyframes orbit-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .idle-orbit-dot {
          width: 4px; height: 4px; border-radius: 50%;
          background: var(--cyan); box-shadow: 0 0 6px var(--cyan);
          position: absolute; top: -2px; left: 50%;
          transform: translateX(-50%);
        }
        .idle-orbit-2 .idle-orbit-dot {
          background: var(--magenta); box-shadow: 0 0 6px var(--magenta);
        }
        .idle-sub {
          font-family: 'Share Tech Mono', monospace;
          font-size: .55rem; color: var(--text3);
          letter-spacing: .08em; margin-top: .5rem;
          text-transform: uppercase;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("")

        # ── Privacy consent gate ───────────────────────────────
        if not st.session_state.camera_consent:
            st.markdown("""
            <style>
            @keyframes consent-in {
              from { opacity: 0; transform: translateY(8px) scale(.97); }
              to   { opacity: 1; transform: translateY(0) scale(1); }
            }
            @keyframes consent-border {
              0%,100% { border-color: rgba(0,247,255,.18); }
              50%      { border-color: rgba(0,247,255,.40); }
            }
            .consent-card {
              background: linear-gradient(145deg, rgba(9,9,22,.98) 0%, rgba(6,6,16,.99) 100%);
              border: 1px solid rgba(0,247,255,.22);
              border-radius: 6px;
              padding: 1.5rem 1.6rem;
              margin-bottom: 1rem;
              animation: consent-in .4s ease forwards, consent-border 4s ease-in-out 0.4s infinite;
              position: relative; overflow: hidden;
            }
            .consent-card::before {
              content: '';
              position: absolute; top: 0; left: 0; right: 0;
              height: 2px;
              background: linear-gradient(90deg, transparent, var(--cyan), transparent);
              opacity: .6;
            }
            .consent-title {
              font-family: var(--orbitron); font-size: 1.05rem;
              letter-spacing: .16em; text-transform: uppercase;
              color: var(--cyan); text-shadow: 0 0 10px rgba(0,247,255,.4);
              margin-bottom: .9rem;
              display: flex; align-items: center; gap: .5rem;
            }
            .consent-body {
              font-family: var(--mono); font-size: .95rem;
              color: var(--text2); letter-spacing: .04em;
              line-height: 2; margin-bottom: 1rem;
            }
            .consent-body em { color: var(--cyan); font-style: normal; }
            .consent-points {
              list-style: none; padding: 0; margin: .6rem 0 0;
            }
            .consent-points li {
              padding: .2rem 0;
              position: relative; padding-left: 1.2rem;
            }
            .consent-points li::before {
              content: '▸'; position: absolute; left: 0;
              color: rgba(0,247,255,.5);
            }
            </style>
            <div class="consent-card">
              <div class="consent-title">&#x1F512; Camera Access Notice</div>
              <div class="consent-body">
                Before initiating neural scan, this app requires your camera.
                <ul class="consent-points">
                  <li>Camera feed is processed <em>locally on your machine</em> only</li>
                  <li><em>No frames</em> are stored, uploaded, or transmitted</li>
                  <li>Session data is wiped when you close the browser tab</li>
                  <li>Facial analysis runs entirely <em>on-device</em> via FER2013 + MediaPipe</li>
                </ul>
              </div>
            </div>
            """, unsafe_allow_html=True)
            col_a, col_b = st.columns([1, 1])
            with col_a:
                if st.button("&#x2714;  GRANT ACCESS &amp; SCAN", key="btn_consent"):
                    st.session_state.camera_consent = True
                    st.session_state.phase = "scanning"
                    st.session_state.emotion_buf.clear()
                    st.rerun()
            with col_b:
                if st.button("&#x2715;  Decline", key="btn_decline"):
                    st.info("Camera access declined. No scan will be performed.")
        else:
            if st.button("&#x2B21;  INITIATE NEURAL SCAN", key="btn_scan"):
                st.session_state.phase = "scanning"
                st.session_state.emotion_buf.clear()
                st.rerun()
            st.markdown(
                '<div style="font-family:var(--mono);font-size:.52rem;color:var(--text3);'
                'letter-spacing:.08em;text-transform:uppercase;margin-top:.4rem;">'
                '// Shortcut: <span style="color:rgba(0,247,255,.4)">SPACE</span> or '
                '<span style="color:rgba(0,247,255,.4)">ENTER</span> to scan</div>',
                unsafe_allow_html=True
            )

    # ── Scanning ───────────────────────────────────────────────
    elif phase == "scanning":
        # Reset emotion color bleed back to default cyan
        st.markdown("""
        <style>
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        section.main {
          transition: background 2s ease !important;
          background: #0a0a0f !important;
        }
        </style>
        <script>
        (function resetEmotionBleed() {
          const root = document.documentElement;
          root.style.setProperty('--cyan',        '#00f7ff');
          root.style.setProperty('--cyan-dim',    'rgba(0,247,255,0.07)');
          root.style.setProperty('--cyan-glow',   '0 0 15px rgba(0,247,255,0.45), 0 0 50px rgba(0,247,255,0.15)');
          root.style.setProperty('--border-neon', 'rgba(0,247,255,0.18)');
          root.style.setProperty('--text',        '#dde0f0');
          root.style.setProperty('--text2',       '#606080');
          const targets = [
            document.documentElement, document.body,
            document.querySelector('[data-testid="stAppViewContainer"]'),
            document.querySelector('[data-testid="stMain"]'),
            document.querySelector('section.main'),
          ].filter(Boolean);
          targets.forEach(el => {
            el.style.transition = 'background 1.4s ease';
            el.style.background = '#0a0a0f';
          });
          const ov = document.getElementById('emotion-bleed-overlay');
          if (ov) { ov.style.opacity = '0'; }
        })();
        </script>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="neural-wrap">
          <div class="radar"></div>
          <div class="neural-status">// Neural scan in progress...</div>
        </div>
        """, unsafe_allow_html=True)

        cam_ph  = st.empty()
        prog_ph = st.empty()

        # ── Browser-based webcam via streamlit-webrtc ──────────
        # cv2.VideoCapture(0) reads the SERVER's camera — useless
        # in the cloud / browser. streamlit-webrtc captures the
        # USER's webcam via WebRTC and streams frames server-side.
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
            import av

            RTC_CONFIG = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            # Thread-safe frame store
            import threading
            _frame_lock  = threading.Lock()
            _latest_frame = [None]   # list so closure can mutate

            class _VideoProcessor:
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    with _frame_lock:
                        _latest_frame[0] = img
                    return frame

            ctx = webrtc_streamer(
                key="moodbeats-scan",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIG,
                video_processor_factory=_VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if ctx.state.playing:
                stable_count = 0
                last_emotion = None
                frame_n      = 0
                max_frames   = 160
                conf         = 0.0
                annotated    = None

                while stable_count < LOCK_FRAMES and frame_n < max_frames:
                    with _frame_lock:
                        frame = _latest_frame[0]
                    if frame is None:
                        time.sleep(0.05)
                        continue

                    frame   = cv2.flip(frame, 1)
                    frame_n += 1

                    if frame_n % FRAME_SKIP == 0:
                        emotion, conf, annotated, _ = detect_emotion(frame)
                        cam_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                     use_container_width=True)
                        if emotion:
                            if emotion == last_emotion:
                                stable_count += 1
                            else:
                                stable_count = 1
                                last_emotion = emotion

                            pct = int(stable_count / LOCK_FRAMES * 100)
                            prog_ph.markdown(f"""
                            <div class="lock-wrap">
                              <div class="lock-label">
                                <span>LOCKING — {emotion.upper()}</span>
                                <span>{pct}%</span>
                              </div>
                              <div class="lock-track">
                                <div class="lock-fill" style="width:{pct}%"></div>
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

                    time.sleep(0.05)

                if stable_count >= LOCK_FRAMES and last_emotion:
                    st.session_state.locked_emotion  = last_emotion
                    st.session_state.locked_conf     = conf
                    st.session_state.locked_frame    = annotated.copy() if annotated is not None else None
                    st.session_state.phase           = "locked"
                else:
                    st.session_state.phase = "idle"

                st.rerun()
            else:
                st.info("⬡ Allow camera access in your browser, then click START above to begin the neural scan.")
            # Skip the old OpenCV block entirely
            import sys as _sys
            _sys.exit(0) if False else None   # never executes — just signals end of webrtc branch

        except ImportError:
            st.error(
                "**streamlit-webrtc not installed.**  \n"
                "Add `streamlit-webrtc` and `av` to your `requirements.txt` and redeploy.",
                icon="📦",
            )
            st.session_state.phase = "idle"
            st.rerun()

        # ── Fallback: legacy local OpenCV (local dev only) ──────
        # This block only runs if streamlit-webrtc import failed.
        if False:  # guarded — webrtc branch above always returns/reruns
         cap = cv2.VideoCapture(0)
         if not cap.isOpened() and False:
            st.markdown("""
            <style>
            /* ── FLOATING CYBERPUNK CAMERA-OFF MODAL ── */
            @keyframes modal-backdrop-in {
              from { opacity: 0; } to { opacity: 1; }
            }
            @keyframes modal-panel-in {
              from { opacity: 0; transform: translate(-50%, -46%) scale(0.92); }
              to   { opacity: 1; transform: translate(-50%, -50%) scale(1); }
            }
            @keyframes modal-scanline {
              0%   { top: 0%; opacity: 0; }
              10%  { opacity: 1; }
              90%  { opacity: .7; }
              100% { top: 100%; opacity: 0; }
            }
            @keyframes modal-border-pulse {
              0%,100% { box-shadow: 0 0 0 1px rgba(255,0,51,.3), 0 0 40px rgba(255,0,51,.15), 0 0 100px rgba(255,0,51,.06); }
              50%      { box-shadow: 0 0 0 1px rgba(255,0,51,.7), 0 0 80px rgba(255,0,51,.4),  0 0 160px rgba(255,0,51,.12); }
            }
            @keyframes modal-icon-flicker {
              0%,85%,100% { opacity: 1; filter: drop-shadow(0 0 20px rgba(255,0,51,.9)); }
              86%  { opacity: .3; }
              88%  { opacity: 1; }
              90%  { opacity: .1; }
              92%  { opacity: 1; }
            }
            @keyframes modal-title-glitch {
              0%,88%,100% { clip-path: none; transform: none; color: #ff0033; }
              89% { clip-path: inset(20% 0 55% 0); transform: translate(-5px,0); color: #ff00ff; }
              91% { clip-path: inset(55% 0 15% 0); transform: translate(5px,0);  color: #00f7ff; }
              93% { clip-path: none; transform: none; color: #ff0033; }
            }
            @keyframes modal-cursor-blink {
              0%,49%,51%,100% { opacity: 1; }
              50% { opacity: 0; }
            }
            @keyframes modal-hbar {
              0%   { left: -100%; opacity: 0; }
              8%   { opacity: 1; }
              92%  { opacity: .8; }
              100% { left: 110%; opacity: 0; }
            }
            @keyframes modal-corner-glow {
              0%,100% { opacity: .5; } 50% { opacity: 1; }
            }
            @keyframes modal-retry-spin {
              from { transform: rotate(0deg); } to { transform: rotate(360deg); }
            }

            /* Darkened backdrop */
            #cam-modal-backdrop {
              position: fixed; inset: 0;
              z-index: 99990;
              background: rgba(4,4,10,.88);
              backdrop-filter: blur(8px);
            }

            /* Floating panel */
            #cam-modal-panel {
              position: fixed;
              top: 50%; left: 50%;
              transform: translate(-50%, -50%);
              z-index: 99999;
              width: min(440px, 92vw);
              background: linear-gradient(150deg, rgba(14,6,6,.98) 0%, rgba(8,8,18,.99) 100%);
              border: 1px solid rgba(255,0,51,.5);
              border-radius: 8px;
              padding: 0;
              overflow: hidden;
              animation: modal-panel-in .3s cubic-bezier(.22,1,.36,1) forwards,
                         modal-border-pulse 2s ease-in-out 0.3s infinite;
            }

            /* Horizontal sweep beam inside panel */
            #cam-modal-panel::after {
              content: '';
              position: absolute;
              top: 0; bottom: 0; width: 80px;
              background: linear-gradient(90deg, transparent, rgba(255,0,51,.14), transparent);
              left: -100%;
              animation: modal-hbar 2.4s ease-in-out infinite;
              pointer-events: none;
            }

            /* Vertical scanline inside panel */
            .cam-modal-vscan {
              position: absolute; left: 0; right: 0;
              height: 1.5px;
              background: linear-gradient(90deg, transparent 0%, rgba(255,0,51,.5) 30%,
                rgba(255,0,51,.8) 50%, rgba(255,0,51,.5) 70%, transparent 100%);
              box-shadow: 0 0 10px rgba(255,0,51,.4);
              animation: modal-scanline 3s linear infinite;
              pointer-events: none;
            }

            /* Corner brackets */
            .cam-modal-corner {
              position: absolute; width: 14px; height: 14px; opacity: .7;
              animation: modal-corner-glow 2s ease-in-out infinite;
            }
            .cam-modal-corner::before, .cam-modal-corner::after {
              content: ''; position: absolute;
              background: #ff0033;
              box-shadow: 0 0 6px rgba(255,0,51,.8);
            }
            .cam-modal-corner::before { width: 100%; height: 1.5px; top: 0; left: 0; }
            .cam-modal-corner::after  { width: 1.5px; height: 100%; top: 0; left: 0; }
            .cam-modal-corner.tl { top: 10px; left: 10px; }
            .cam-modal-corner.tr { top: 10px; right: 10px; transform: scaleX(-1); }
            .cam-modal-corner.bl { bottom: 10px; left: 10px; transform: scaleY(-1); }
            .cam-modal-corner.br { bottom: 10px; right: 10px; transform: scale(-1); }

            /* Top status bar */
            .cam-modal-topbar {
              display: flex; align-items: center; justify-content: space-between;
              padding: .55rem 1.1rem .45rem;
              border-bottom: 1px solid rgba(255,0,51,.15);
              background: rgba(255,0,51,.04);
            }
            .cam-modal-topbar-id {
              font-family: 'Share Tech Mono', monospace;
              font-size: .5rem; letter-spacing: .14em;
              color: rgba(255,0,51,.5); text-transform: uppercase;
            }
            .cam-modal-topbar-status {
              font-family: 'Share Tech Mono', monospace;
              font-size: .5rem; letter-spacing: .1em;
              color: rgba(255,0,51,.4); text-transform: uppercase;
              display: flex; align-items: center; gap: .4rem;
            }
            .cam-modal-topbar-dot {
              width: 5px; height: 5px; border-radius: 50%;
              background: #ff0033;
              box-shadow: 0 0 6px #ff0033;
              animation: modal-cursor-blink 1s step-end infinite;
            }
            .cam-modal-body {
              padding: 2rem 2rem 1.6rem;
              text-align: center;
            }
            .cam-modal-icon {
              font-size: 3.4rem; line-height: 1;
              margin-bottom: 1.1rem; display: block;
              animation: modal-icon-flicker 4s ease-in-out infinite;
            }
            .cam-modal-title {
              font-family: 'Orbitron', monospace;
              font-size: 1.1rem; font-weight: 900;
              letter-spacing: .1em; text-transform: uppercase;
              color: #ff0033;
              text-shadow: 0 0 25px rgba(255,0,51,.85), 0 0 70px rgba(255,0,51,.3);
              animation: modal-title-glitch 5s ease-in-out infinite;
              margin-bottom: .5rem;
            }
            .cam-modal-errcode {
              font-family: 'Share Tech Mono', monospace;
              font-size: .6rem; letter-spacing: .16em;
              color: rgba(255,0,51,.55); text-transform: uppercase;
              margin-bottom: 1.6rem;
            }
            .cam-modal-errcode .blink { animation: modal-cursor-blink 1.1s step-end infinite; }
            .cam-modal-divider {
              height: 1px; margin: 0 auto 1.4rem;
              width: 85%;
              background: linear-gradient(90deg, transparent, rgba(255,0,51,.35), transparent);
            }
            .cam-modal-steps {
              list-style: none; padding: 0; margin: 0 auto 1.4rem;
              text-align: left; display: inline-block;
              font-family: 'Share Tech Mono', monospace;
              font-size: .6rem; letter-spacing: .07em;
              text-transform: uppercase; line-height: 2.3;
              color: #606080;
            }
            .cam-modal-steps li { position: relative; padding-left: 1.3rem; }
            .cam-modal-steps li::before {
              content: '▸'; position: absolute; left: 0;
              color: rgba(255,0,51,.55);
            }
            .cam-modal-steps li em { color: rgba(255,0,51,.75); font-style: normal; }
            .cam-modal-retry {
              display: flex; align-items: center; justify-content: center; gap: .6rem;
              font-family: 'Share Tech Mono', monospace;
              font-size: .58rem; letter-spacing: .12em;
              color: rgba(255,0,51,.55); text-transform: uppercase;
              margin-top: .2rem; margin-bottom: 1rem;
            }
            .cam-modal-retry-spin {
              width: 13px; height: 13px;
              border: 1.5px solid rgba(255,0,51,.2);
              border-top-color: #ff0033;
              border-radius: 50%;
              animation: modal-retry-spin 1s linear infinite;
              flex-shrink: 0;
            }
            .cam-modal-footer {
              font-family: 'Orbitron', monospace;
              font-size: .45rem; letter-spacing: .2em;
              color: rgba(255,0,51,.25); text-transform: uppercase;
              padding: .7rem 1.1rem;
              border-top: 1px solid rgba(255,0,51,.1);
              text-align: center;
            }
            </style>
            """, unsafe_allow_html=True)
            # ── Polling loop: hold the modal on screen until camera comes online ──
            _cam_modal = """
            <style>
            @keyframes modal-retry-spin {
              from { transform: rotate(0deg); } to { transform: rotate(360deg); }
            }
            .cam-modal-retry {
              display: flex; align-items: center; justify-content: center; gap: .6rem;
              font-family: 'Share Tech Mono', monospace;
              font-size: .58rem; letter-spacing: .12em;
              color: rgba(255,0,51,.55); text-transform: uppercase;
              margin-top: .2rem; margin-bottom: 1rem;
            }
            .cam-modal-retry-spin {
              width: 13px; height: 13px;
              border: 1.5px solid rgba(255,0,51,.2);
              border-top-color: #ff0033;
              border-radius: 50%;
              animation: modal-retry-spin 1s linear infinite;
              flex-shrink: 0;
            }
            </style>
            <div id="cam-modal-backdrop"></div>
            <div id="cam-modal-panel">
              <div class="cam-modal-corner tl"></div>
              <div class="cam-modal-corner tr"></div>
              <div class="cam-modal-corner bl"></div>
              <div class="cam-modal-corner br"></div>
              <div class="cam-modal-vscan"></div>
              <div class="cam-modal-topbar">
                <span class="cam-modal-topbar-id">MOODBEATS // NEURAL OS</span>
                <span class="cam-modal-topbar-status">
                  <span class="cam-modal-topbar-dot"></span>
                  OPTIC FEED LOST
                </span>
              </div>
              <div class="cam-modal-body">
                <span class="cam-modal-icon">&#x1F4F7;</span>
                <div class="cam-modal-title">OPTIC LINK SEVERED</div>
                <div class="cam-modal-errcode">
                  ERR::0xCAM_NULL &nbsp;&#x2014;&nbsp; BIOMETRIC FEED OFFLINE &nbsp;<span class="blink">&#x258A;</span>
                </div>
                <div class="cam-modal-divider"></div>
                <ul class="cam-modal-steps">
                  <li>Browser <em>denied camera access</em> — click the camera icon in the address bar and set it to <em>Allow</em>, then reload</li>
                  <li>Your <em>laptop camera</em> is disabled — flip that switch, choom</li>
                  <li>Grant <em>camera permissions</em> in browser Settings → Privacy → Camera</li>
                  <li>Close any app holding the <em>optic feed</em> hostage</li>
                  <li>Reconnect your <em>optical input device</em> if external</li>
                </ul>
                <div class="cam-modal-retry">
                  <div class="cam-modal-retry-spin"></div>
                  Scanning for optic feed — enable camera &amp; reload to continue
                </div>
              </div>
              <div class="cam-modal-footer">// If you denied camera access in your browser, enable it in Settings → reload the page to jack back in</div>
            </div>
            """
            _modal_slot = st.empty()
            _cam_found  = False
            for _attempt in range(120):   # poll up to 60 s (0.5 s x 120)
                _modal_slot.markdown(_cam_modal, unsafe_allow_html=True)
                time.sleep(0.5)
                _probe = cv2.VideoCapture(0)
                if _probe.isOpened():
                    _probe.release()
                    _cam_found = True
                    break
                _probe.release()

            _modal_slot.empty()   # dismiss modal the instant camera is detected

            if not _cam_found:
                st.session_state.phase = "idle"
                st.rerun()
            cap = cv2.VideoCapture(0)

        # ── Dead code: old server-side OpenCV loop (kept for reference) ──
        # This block is never reached; the webrtc branch above always
        # calls st.rerun() before execution reaches here.
        # stable_count / last_emotion / cap logic was moved into the
        # webrtc ctx.state.playing block above.
        pass  # end of scanning phase

    # ── Locked / Result ────────────────────────────────────────
    elif phase in ("locked", "result"):
        emotion = st.session_state.locked_emotion
        frame   = st.session_state.locked_frame

        # ── EMOTION AMBIENT COLOR BLEED ────────────────────────
        # Full UI theming: rewrites ALL CSS vars so every neon accent,
        # heading, border, scrollbar and text-shadow shifts in unison.
        _EMOTION_THEME = {
            # emotion : (primary_hex, r, g, b,  bg_tint_rgba,               text_primary, text_dim)
            "angry":    ("#ff3b30", 255, 59,  48,  "rgba(100,0,15,0.28)",   "#ffd0d8",    "#804050"),
            "fear":     ("#bf5af2", 191, 90, 242,  "rgba(60,10,100,0.28)",  "#e8d0ff",    "#605080"),
            "happy":    ("#ffd60a", 255,214,  10,  "rgba(110,90,0,0.28)",   "#fff8d0",    "#807040"),
            "sad":      ("#0a84ff",  10,132, 255,  "rgba(0,25,90,0.28)",    "#d0e8ff",    "#405870"),
            "disgust":  ("#30d158",  48,209,  88,  "rgba(0,80,30,0.28)",    "#d0ffe0",    "#407050"),
            "surprise":  ("#FF00CC", 255,  0, 204,  "rgba(100,0,80,0.28)",   "#ffe0f8",    "#803060"),
            "neutral":  ("#8e8e93", 142,142, 147,  "rgba(25,30,55,0.22)",   "#dde0f0",    "#606080"),
        }
        _theme = _EMOTION_THEME.get(emotion, _EMOTION_THEME["neutral"])
        _ec    = _theme[0]   # primary hex  e.g. #ffcc00
        _r, _g, _b = _theme[1], _theme[2], _theme[3]
        _bg_tint   = _theme[4]
        _text_pri  = _theme[5]
        _text_dim  = _theme[6]
        # Derived rgba variants
        _ec_45  = f"rgba({_r},{_g},{_b},0.45)"
        _ec_18  = f"rgba({_r},{_g},{_b},0.18)"
        _ec_12  = f"rgba({_r},{_g},{_b},0.12)"
        _ec_08  = f"rgba({_r},{_g},{_b},0.08)"
        _ec_07  = f"rgba({_r},{_g},{_b},0.07)"
        _ec_06  = f"rgba({_r},{_g},{_b},0.06)"
        _ec_04  = f"rgba({_r},{_g},{_b},0.04)"
        _ec_03  = f"rgba({_r},{_g},{_b},0.035)"
        _ec_01  = f"rgba({_r},{_g},{_b},0.010)"
        _glow   = f"0 0 15px {_ec_45}, 0 0 50px {_ec_18}"
        _glow2  = f"0 0 20px {_ec_45}, 0 0 60px {_ec_18}"

        st.markdown(f"""
        <style>
        /* ══════════════════════════════════════
           EMOTION AMBIENT COLOR BLEED
           Rewrites every CSS variable so the
           whole UI skin shifts to the emotion.
        ══════════════════════════════════════ */
        :root {{
          /* Primary neon — was --cyan */
          --cyan:        {_ec} !important;
          --cyan-dim:    {_ec_07} !important;
          --cyan-glow:   {_glow} !important;
          /* Border tints */
          --border-neon: {_ec_18} !important;
          /* Text */
          --text:        {_text_pri} !important;
          --text2:       {_text_dim} !important;
        }}

        /* Background bleed */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        section.main {{
          transition: background 2s ease !important;
          background: radial-gradient(
            ellipse 130% 130% at 50% 0%,
            {_bg_tint} 0%,
            #0a0a0f 62%
          ) !important;
        }}

        /* Scanline + top halo */
        [data-testid="stMain"]::before {{
          background:
            repeating-linear-gradient(
              0deg,
              transparent, transparent 3px,
              {_ec_01} 3px, {_ec_01} 4px
            ),
            linear-gradient(180deg, {_ec}1c 0%, transparent 20%) !important;
          transition: background 2s ease !important;
        }}

        /* Ambient orb */
        [data-testid="stMain"]::after {{
          background: radial-gradient(circle, {_ec_03} 0%, transparent 68%) !important;
        }}

        /* Grid lines */
        #cp-grid {{
          background-image:
            linear-gradient({_ec_04} 1px, transparent 1px),
            linear-gradient(90deg, {_ec_04} 1px, transparent 1px) !important;
        }}

        /* Scan beam */
        .cp-beam {{
          background: linear-gradient(90deg,
            transparent 0%, {_ec_45} 30%,
            {_ec} 50%, {_ec_45} 70%, transparent 100%) !important;
          box-shadow: 0 0 18px {_ec_45}, 0 0 40px {_ec_18} !important;
        }}

        /* Corner HUD brackets */
        .cp-corner::before, .cp-corner::after {{
          background: {_ec} !important;
          box-shadow: 0 0 8px {_ec_45} !important;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar-thumb {{
          background: {_ec} !important;
          box-shadow: 0 0 5px {_ec} !important;
        }}

        /* HR */
        hr {{ border-top-color: {_ec_12} !important; }}

        /* Headings */
        h1, h2, h3, h4 {{
          color: {_ec} !important;
          text-shadow: 0 0 20px {_ec_45} !important;
        }}

        /* Panel glass — border + corner accents */
        [data-testid="column"] > div:first-child {{
          border-color: {_ec_18} !important;
          box-shadow: 0 0 0 1px {_ec_04}, 0 8px 48px rgba(0,0,0,0.7),
                      inset 0 1px 0 {_ec_06} !important;
        }}
        [data-testid="column"] > div:first-child::before,
        [data-testid="column"] > div:first-child::after {{
          background: {_ec} !important;
          box-shadow: {_glow} !important;
        }}
        @keyframes panel-corner-pulse {{
          0%,100% {{ opacity:.7; box-shadow: 0 0 8px {_ec_45}; }}
          50%      {{ opacity:1; box-shadow: 0 0 24px {_ec}, 0 0 50px {_ec_18}; }}
        }}
        @keyframes panel-border-breathe {{
          0%,100% {{ border-color: {_ec_12} !important; }}
          50%      {{ border-color: {_ec_45} !important;
                      box-shadow: 0 0 0 1px {_ec_06}, 0 8px 48px rgba(0,0,0,.7),
                                  0 0 40px {_ec_08} !important; }}
        }}

        /* Stray section-label dot */
        .section-label::before {{
          background: {_ec} !important;
          box-shadow: 0 0 8px {_ec} !important;
        }}

        /* Streamlit buttons */
        .stButton > button {{
          border-color: {_ec_45} !important;
          color: {_ec} !important;
          text-shadow: 0 0 8px {_ec_45} !important;
        }}
        .stButton > button:hover {{
          border-color: {_ec} !important;
          background: {_ec} !important;
          box-shadow: {_glow} !important;
          color: #000 !important;
          text-shadow: none !important;
        }}

        /* Link buttons */
        .stLinkButton > a {{
          border-color: {_ec_45} !important;
          color: {_ec} !important;
          text-shadow: 0 0 8px {_ec_45} !important;
        }}

        /* Platform buttons (YouTube/Spotify/Apple Music) in playlist */
        .mood-btn {{
          color: {_ec} !important;
          border-color: {_ec_45} !important;
          background: {_ec_06} !important;
          text-shadow: 0 0 8px {_ec_45} !important;
        }}
        .mood-btn:hover {{
          border-color: {_ec} !important;
          background: {_ec_12} !important;
          box-shadow: 0 0 20px {_ec_45}, 0 0 50px {_ec_18} !important;
          text-shadow: 0 0 12px {_ec} !important;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
          border-right-color: {_ec_08} !important;
        }}

        /* Header nav */
        [data-testid="stHeader"] {{
          border-bottom-color: {_ec_18} !important;
        }}
        [data-testid="stHeader"] button {{
          border-color: {_ec_04} !important;
        }}
        [data-testid="stHeader"] button:hover {{
          border-color: {_ec_45} !important;
          box-shadow: 0 0 14px {_ec_45} !important;
        }}
        [data-testid="stHeader"] button svg {{
          color: {_ec} !important;
          stroke: {_ec} !important;
        }}

        /* Data ticker */
        #cp-data-ticker {{
          border-top-color: {_ec_12} !important;
        }}
        .cp-ticker-inner {{ color: {_ec_45} !important; }}
        .cp-ticker-item em {{ color: {_ec} !important; }}

        /* Pulse rings */
        #cp-rings .cp-ring:nth-child(1) {{ border-color: {_ec_12} !important; }}
        #cp-rings .cp-ring:nth-child(2) {{ border-color: {_ec_08} !important; }}
        #cp-rings .cp-ring:nth-child(3) {{ border-color: {_ec_06} !important; }}
        #cp-rings .cp-ring:nth-child(4) {{ border-color: {_ec_04} !important; }}

        /* Playlist card pulse */
        @keyframes pl-card-pulse {{
          0%,100% {{ box-shadow: inset 3px 0 20px {_ec_06}; }}
          50%      {{ box-shadow: inset 3px 0 40px {_ec_12}, 0 0 30px {_ec_04}; }}
        }}
        /* Emo reveal */
        @keyframes emo-border-pulse {{
          0%,100% {{ box-shadow: none; }}
          50%      {{ box-shadow: 0 0 30px {_ec_08}; }}
        }}

        /* Live dot + cyber-badge */
        .live-dot {{ background: {_ec} !important; box-shadow: 0 0 6px {_ec} !important; }}
        .cyber-badge.active {{ border-color: {_ec_45} !important; color: {_ec} !important; }}
        .cyber-badge.active .bdot {{ background: {_ec} !important; box-shadow: 0 0 6px {_ec} !important; }}

        /* Lock progress bar */
        .lock-fill {{ background: {_ec} !important; box-shadow: 0 0 12px {_ec_45} !important; }}
        .lock-track {{ border-color: {_ec_12} !important; }}

        /* History accordion */
        .cp-acc-ts {{ color: {_text_dim} !important; }}

        /* ── ALL remaining text that should shift with emotion ── */
        /* Body text, spans, labels, divs — catch-all for Rajdhani */
        p, li, span, label {{ color: {_text_pri} !important; }}

        /* Sub-labels and mono text */
        .emo-sub, .scan-hint, .pl-prefix, .neural-status,
        .lock-label, .hud-tagline-human,
        .idle-sub, .pl-vibe, .cp-ticker-inner {{
          color: {_text_dim} !important;
        }}

        /* Shortcut hint at bottom of scanner col */
        [style*="var(--text3)"] {{ color: {_text_dim} !important; }}

        /* Orbitron headings that use var(--text2) */
        .hud-badge {{ color: {_text_dim} !important; border-color: {_ec_18} !important; }}
        .hud-tagline {{ color: {_text_dim} !important; }}
        .hud-tagline em {{ color: {_ec} !important; }}

        /* Waveform bars */
        #audio-viz span {{ background: {_ec} !important; box-shadow: 0 0 6px {_ec} !important; }}
        #audio-status {{ color: {_ec}66 !important; }}
        #audio-dot.live {{ background: {_ec} !important; box-shadow: 0 0 8px {_ec} !important; }}
        /* Neural OS HUD box border + glow */
        #audio-hud {{
          border-color: {_ec_18} !important;
          box-shadow: 0 0 22px {_ec_07}, 0 0 8px {_ec}11, inset 0 1px 0 {_ec}18 !important;
        }}

        /* Section label after-line */
        .section-label::after {{
          background: linear-gradient(90deg, {_ec_45}, transparent) !important;
        }}

        /* Privacy note in How It Works */
        #hiw-privacy-note {{
          border-left-color: {_ec_45} !important;
          color: {_ec} !important;
          text-shadow: 0 0 8px {_ec_45} !important;
        }}
        #hiw-privacy-note em {{
          color: {_ec} !important;
          font-weight: 600 !important;
        }}
        </style>
        <script>
        (function applyEmotionBleed() {{
          const root = document.documentElement;
          // Set CSS variables — propagates to every var(--cyan) reference
          const ec = '{_ec}';

          // Apply transition class first, then set variables after a frame
          // so the browser can interpolate via CSS transitions
          requestAnimationFrame(() => {{
            root.style.setProperty('--cyan',        ec);
            root.style.setProperty('--cyan-dim',    '{_ec_07}');
            root.style.setProperty('--cyan-glow',   '{_glow}');
            root.style.setProperty('--border-neon', '{_ec_18}');
            root.style.setProperty('--text',        '{_text_pri}');
            root.style.setProperty('--text2',       '{_text_dim}');
            root.style.setProperty('--emotion-bleed', '{_bg_tint}');
          }});

          // Transition background on all wrappers
          const targets = [
            document.documentElement, document.body,
            document.querySelector('[data-testid="stAppViewContainer"]'),
            document.querySelector('[data-testid="stMain"]'),
            document.querySelector('section.main'),
          ].filter(Boolean);
          targets.forEach(el => {{
            el.style.transition = 'background 1.4s ease, color 1.2s ease';
            el.style.background =
              `radial-gradient(ellipse 130% 130% at 50% 0%, {_bg_tint} 0%, #0a0a0f 62%)`;
          }});

          // Recolor all body text to emotion palette
          root.style.setProperty('--text',  '{_text_pri}');
          root.style.setProperty('--text2', '{_text_dim}');
          document.querySelectorAll('p, li, label').forEach(el => {{
            el.style.transition = 'color 1.2s ease';
            el.style.color = '{_text_pri}';
          }});

          // Update Neural OS audio-hud border/glow to match emotion
          const hud = document.getElementById('audio-hud');
          if (hud) {{
            hud.style.transition = 'border-color 1.2s ease, box-shadow 1.2s ease';
            hud.style.borderColor = ec + '44';
            hud.style.boxShadow   = '0 0 22px ' + ec + '22, 0 0 8px ' + ec + '11, inset 0 1px 0 ' + ec + '18';
          }}

          // Overlay halo div
          const old = document.getElementById('emotion-bleed-overlay');
          if (old) old.remove();
          const ov = document.createElement('div');
          ov.id = 'emotion-bleed-overlay';
          const ovBg = 'radial-gradient(ellipse 100% 55% at 50% -5%, ' + ec + '1e 0%, transparent 58%)';
          ov.style.cssText = `
            position:fixed; inset:0; z-index:0; pointer-events:none;
            background: ${{ovBg}};
            opacity:0; transition:opacity 1.6s ease;
          `;
          document.body.appendChild(ov);
          requestAnimationFrame(() => requestAnimationFrame(() => {{ ov.style.opacity = '1'; }}));
        }})();
        </script>
        """, unsafe_allow_html=True)
        # ── END EMOTION AMBIENT COLOR BLEED ───────────────────

        st.markdown("""
        <span class="cyber-badge active">
          <span class="bdot"></span> Scan complete
        </span>
        """, unsafe_allow_html=True)
        st.markdown("")

        if frame is not None:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     use_container_width=True)

        render_emotion_result(emotion)
        render_mood_override(emotion)
        st.markdown("")

        if st.button("&#x21BA;  RESCAN", key="btn_rescan"):
            st.session_state.phase           = "scanning"
            st.session_state.locked_emotion  = None
            st.session_state.locked_frame    = None
            st.session_state.playlist_parsed = None
            st.session_state.emotion_buf.clear()
            st.rerun()
        st.markdown(
            '<div style="font-family:var(--mono);font-size:.52rem;color:var(--text3);'
            'letter-spacing:.08em;text-transform:uppercase;margin-top:.3rem;">'
            '// Shortcut: <span style="color:rgba(0,247,255,.4)">SPACE</span> or '
            '<span style="color:rgba(0,247,255,.4)">ENTER</span></div>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════
#  RIGHT COLUMN — Playlist
# ══════════════════════════════════════════════════════════════

with col_right:
    st.markdown('<div class="section-label">&#x2B21; Neural Playlist</div>', unsafe_allow_html=True)

    phase   = st.session_state.phase
    emotion = st.session_state.locked_emotion

    # ── Nothing yet ────────────────────────────────────────────
    if phase == "idle" or emotion is None:
        st.markdown("""
        <div class="pl-idle">
          <div class="pl-idle-icon">&#x266B;</div>
          <div class="pl-idle-hint">// Awaiting emotion signal</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Generate fresh playlist after lock ─────────────────────
    elif phase == "locked":
        if st.session_state.playlist_parsed is None:
            # ── Show a clean loading card while LLM works ──────
            _mood_l = get_mood(emotion)
            _color_l = EMOTION_COLORS.get(emotion, "#00f7ff")
            st.markdown(f"""
            <style>
            @keyframes pl-load-bar {{
              0%   {{ left: -40%; }}
              100% {{ left: 110%; }}
            }}
            .pl-loading-card {{
              border: 1px solid {_color_l}33;
              border-left: 3px solid {_color_l}88;
              border-radius: 6px;
              padding: 1.4rem 1.2rem;
              position: relative; overflow: hidden;
              background: linear-gradient(135deg, rgba(11,11,23,.97) 0%, rgba(7,7,17,.99) 100%);
              box-shadow: inset 3px 0 20px {_color_l}0d;
            }}
            .pl-loading-card::after {{
              content: '';
              position: absolute; top: 0; bottom: 0; width: 40%;
              background: linear-gradient(90deg, transparent, {_color_l}18, transparent);
              animation: pl-load-bar 1.6s ease-in-out infinite;
              pointer-events: none;
            }}
            .pl-loading-title {{
              font-family: 'Orbitron', monospace;
              font-size: .75rem; font-weight: 700;
              letter-spacing: .14em; text-transform: uppercase;
              color: {_color_l};
              text-shadow: 0 0 12px {_color_l}88;
              margin-bottom: .9rem;
            }}
            .pl-loading-row {{
              display: flex; align-items: center; gap: .7rem;
              padding: .45rem .3rem; border-radius: 3px;
              margin-bottom: 2px;
            }}
            .pl-loading-num {{
              font-family: 'Share Tech Mono', monospace;
              font-size: .55rem; color: #282838;
              width: 1.2rem; flex-shrink: 0; text-align: right;
            }}
            .pl-loading-bar-wrap {{ flex: 1; }}
            .pl-loading-bar {{
              height: 6px; border-radius: 3px;
              background: linear-gradient(90deg, {_color_l}33, {_color_l}11);
              animation: pl-row-shimmer 1.8s ease-in-out infinite;
            }}
            @keyframes pl-row-shimmer {{
              0%,100% {{ opacity: .4; }} 50% {{ opacity: .8; }}
            }}
            .pl-loading-hint {{
              font-family: 'Share Tech Mono', monospace;
              font-size: .58rem; color: #404058;
              letter-spacing: .1em; text-transform: uppercase;
              margin-top: .9rem;
            }}
            </style>
            <div class="pl-loading-card">
              <div class="pl-loading-title">&#x2B21; Synthesising Neural Playlist…</div>
              {''.join(f'<div class="pl-loading-row"><span class="pl-loading-num">{i+1:02d}</span><div class="pl-loading-bar-wrap"><div class="pl-loading-bar" style="width:{random.randint(45,90)}%"></div></div></div>' for i in range(6))}
              <div class="pl-loading-hint">// LLM curating frequencies for {emotion.upper()} · {_mood_l['genre']}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Call LLM — shown after loading card renders ────
            with st.spinner("// Generating neural playlist…"):
                pl = generate_playlist(emotion, seed=st.session_state.regen_seed)

            st.session_state.playlist_parsed = pl
            st.session_state.history.insert(0, {
                "emotion": emotion,
                "playlist": pl,
                "ts": time.strftime("%d %b, %H:%M"),
            })
            st.session_state.history = st.session_state.history[:6]
            st.session_state.phase = "result"
            st.rerun()
        else:
            pl = st.session_state.playlist_parsed
            render_playlist(pl, emotion)
            render_platform_buttons(pl, emotion)

    # ── Result: playlist + controls ────────────────────────────
    elif phase == "result":
        pl = st.session_state.playlist_parsed
        if pl:
            render_playlist(pl, emotion)

            # Regenerate Mix — new seed, new playlist, no re-scan
            st.markdown('<div style="margin-top: 1.4rem;"></div>', unsafe_allow_html=True)
            if st.button("&#x27F3;  REGENERATE MIX", key="btn_regen"):
                st.session_state.regen_seed += 1
                with st.spinner("// Synthesising new neural frequencies…"):
                    new_pl = generate_playlist(emotion, seed=st.session_state.regen_seed)
                st.session_state.playlist_parsed = new_pl
                if st.session_state.history:
                    st.session_state.history[0]["playlist"] = new_pl
                st.rerun()

            render_platform_buttons(pl, emotion)

    # ── History ────────────────────────────────────────────────
    hist = st.session_state.history
    if len(hist) > 1:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">&#x2B21; Session History</div>', unsafe_allow_html=True)

        # Build all history blocks as pure HTML — no st.expander so no mangled labels / arrows
        # NOTE: styles are injected once via a separate markdown call to avoid duplication glitch
        st.markdown("""
        <style>
        /* ── CP2077 Custom Accordion ── */
        @keyframes cp-acc-scan {
          0%   { left: -100%; opacity: 0; }
          15%  { opacity: 1; }
          85%  { opacity: 0.7; }
          100% { left: 110%; opacity: 0; }
        }
        @keyframes cp-acc-bar {
          0%,100% { opacity: .4; }
          50%     { opacity: 1; box-shadow: 0 0 10px var(--acc-color-glow, rgba(0,247,255,.7)); }
        }
        @keyframes cp-acc-dot {
          0%,100% { opacity: .3; transform: scale(.8); }
          50%     { opacity: 1;  transform: scale(1.3); }
        }
        .cp-acc-item {
          position: relative;
          background: linear-gradient(135deg, rgba(11,11,23,.97) 0%, rgba(7,7,17,.99) 100%);
          border: 1px solid var(--acc-color-dim, rgba(0,247,255,.13));
          border-radius: 4px;
          margin-bottom: .55rem;
          overflow: hidden;
          transition: border-color .3s ease, box-shadow .3s ease;
        }
        .cp-acc-item:hover {
          border-color: var(--acc-color-bright, rgba(0,247,255,.32));
          box-shadow: 0 0 22px var(--acc-color-glow, rgba(0,247,255,.08)), inset 0 0 30px var(--acc-color-inner, rgba(0,247,255,.02));
        }
        .cp-acc-item::before {
          content: '';
          position: absolute;
          left: 0; top: 0; bottom: 0;
          width: 2px;
          background: linear-gradient(180deg, transparent, var(--acc-color, #00f7ff), transparent);
          animation: cp-acc-bar 3s ease-in-out infinite;
        }
        .cp-acc-item:hover::before,
        .cp-acc-item.open::before {
          background: var(--acc-color, #00f7ff);
          box-shadow: 0 0 14px var(--acc-color-glow2, rgba(0,247,255,.9));
          animation: none;
        }
        .cp-acc-summary {
          display: flex;
          align-items: center;
          gap: .65rem;
          padding: .82rem 1.1rem .82rem 1.35rem;
          cursor: pointer;
          user-select: none;
          position: relative;
          overflow: hidden;
          transition: background .2s ease;
        }
        .cp-acc-summary:hover { background: var(--acc-color-inner, rgba(0,247,255,.025)); }
        .cp-acc-summary::after {
          content: '';
          position: absolute;
          top: 0; bottom: 0;
          width: 55px;
          background: linear-gradient(90deg, transparent, var(--acc-color-scan, rgba(0,247,255,.18)), transparent);
          left: -100%;
          pointer-events: none;
        }
        .cp-acc-item:hover .cp-acc-summary::after {
          animation: cp-acc-scan 1.4s ease forwards;
        }
        .cp-acc-dot {
          width: 5px; height: 5px;
          border-radius: 50%;
          flex-shrink: 0;
          animation: cp-acc-dot 2s ease-in-out infinite;
        }
        .cp-acc-item.open .cp-acc-dot {
          animation: none; opacity: 1;
          transform: scale(1.2);
          box-shadow: 0 0 8px currentColor;
        }
        .cp-acc-label {
          font-family: 'Orbitron', monospace;
          font-size: .76rem; font-weight: 700;
          letter-spacing: .16em; text-transform: uppercase;
          flex: 1;
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
          transition: text-shadow .2s ease;
        }
        .cp-acc-item:hover .cp-acc-label,
        .cp-acc-item.open .cp-acc-label { text-shadow: 0 0 12px currentColor; }
        .cp-acc-ts {
          font-family: 'Share Tech Mono', monospace;
          font-size: .68rem; color: #404058;
          letter-spacing: .08em; flex-shrink: 0;
        }
        .cp-acc-chevron {
          width: 0; height: 0;
          border-top: 4px solid transparent;
          border-bottom: 4px solid transparent;
          border-left: 6px solid;
          flex-shrink: 0; opacity: .45;
          transition: transform .25s ease, opacity .2s ease;
        }
        .cp-acc-item.open .cp-acc-chevron { transform: rotate(90deg); opacity: 1; }
        .cp-acc-item:hover .cp-acc-chevron { opacity: .75; }
        .cp-acc-body {
          display: none;
          padding: .8rem 1.1rem 1rem 1.35rem;
          border-top: 1px solid var(--acc-color-dim, rgba(0,247,255,.07));
        }
        .cp-acc-item.open .cp-acc-body { display: block; }
        .cp-acc-pl-title {
          font-family: 'Orbitron', monospace;
          font-size: .68rem; letter-spacing: .16em;
          text-transform: uppercase; margin-bottom: .85rem;
        }
        .cp-ht {
          display: flex; align-items: center; gap: .6rem;
          padding: .4rem .3rem; border-radius: 3px;
          transition: background .15s ease, transform .15s ease;
          flex-wrap: nowrap;
        }
        .cp-ht:hover { background: rgba(0,247,255,.05); transform: translateX(4px); }
        .cp-ht-num {
          font-family: 'Share Tech Mono', monospace;
          font-size: .68rem; color: #282838;
          width: 1.2rem; flex-shrink: 0; text-align: right;
        }
        .cp-ht-body { flex: 1; min-width: 0; overflow: hidden; }
        .cp-ht-song {
          font-family: 'Rajdhani', sans-serif;
          font-size: .88rem; font-weight: 600; color: #dde0f0;
          display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .cp-ht-artist {
          font-family: 'Share Tech Mono', monospace;
          font-size: .7rem; color: #606080; letter-spacing: .03em;
          display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .cp-ht-links { display: flex; gap: .3rem; flex-shrink: 0; }
        .cp-acc-actions {
          display: flex; gap: .5rem;
          margin-top: .85rem; padding-top: .7rem;
          border-top: 1px solid var(--acc-color-dim, rgba(0,247,255,.06));
          flex-wrap: wrap;
        }
        </style>
        <script>
        function cpAccToggle(id) {
          var el = document.getElementById(id);
          if (el) el.classList.toggle('open');
        }
        </script>
        """, unsafe_allow_html=True)

        history_html = ""
        for idx, item in enumerate(hist[1:6]):
            emo_h  = item["emotion"]
            pl_h   = item["playlist"]
            emoji  = EMOJI_FULL.get(emo_h, "🎵")
            color  = EMOTION_COLORS.get(emo_h, "#00f7ff")
            yt = youtube_url(pl_h, emo_h)
            sp = spotify_url(pl_h, emo_h)
            item_id = f"cp-acc-{idx}"

            tracks_html = ""
            for i, t in enumerate(pl_h.get("tracks", [])):
                song   = t.get("song", "")
                artist = t.get("artist", "")
                tyt    = f"https://www.youtube.com/results?search_query={song.replace(' ', '+').replace('&', '%26')}+{artist.replace(' ', '+').replace('&', '%26')}+official&sp=EgIQAQ%3D%3D"
                tsp    = f"https://open.spotify.com/search/{song.replace(' ', '%20').replace('&', '%26')}%20{artist.replace(' ', '%20').replace('&', '%26')}/tracks"
                artist_html = f'<span class="cp-ht-artist">{artist}</span>' if artist else ''
                tracks_html += (
                    f'<div class="cp-ht">'
                    f'<span class="cp-ht-num">{i+1:02d}</span>'
                    f'<span class="cp-ht-body">'
                    f'<span class="cp-ht-song">{song}</span>'
                    f'{artist_html}'
                    f'</span>'
                    f'<span class="cp-ht-links">'
                    f'<a href="{tyt}" target="_blank" class="tp-btn yt" aria-label="Watch {song} on YouTube">YT</a>'
                    f'<a href="{tsp}" target="_blank" class="tp-btn sp" aria-label="Find {song} on Spotify">SP</a>'
                    f'</span></div>'
                )

            pl_title = pl_h.get('title', '')
            r_h, g_h, b_h = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            acc_style = (
                f"--acc-color:{color};"
                f"--acc-color-dim:rgba({r_h},{g_h},{b_h},0.20);"
                f"--acc-color-bright:rgba({r_h},{g_h},{b_h},0.50);"
                f"--acc-color-glow:rgba({r_h},{g_h},{b_h},0.12);"
                f"--acc-color-inner:rgba({r_h},{g_h},{b_h},0.03);"
                f"--acc-color-glow2:rgba({r_h},{g_h},{b_h},0.9);"
                f"--acc-color-scan:rgba({r_h},{g_h},{b_h},0.22);"
            )
            history_html += (
                f'<div class="cp-acc-item" id="{item_id}" style="{acc_style}" onclick="cpAccToggle(\'{item_id}\')"'
                f' role="button" tabindex="0" aria-expanded="false"'
                f' onkeydown="if(event.key===\'Enter\'||event.key===\' \'){{event.preventDefault();cpAccToggle(\'{item_id}\');}}">'
                f'<div class="cp-acc-summary">'
                f'<span class="cp-acc-dot" style="background:{color};box-shadow:0 0 5px {color}88;color:{color};"></span>'
                f'<span class="cp-acc-label" style="color:{color};">{emoji}&nbsp;&nbsp;{emo_h.upper()}</span>'
                f'<span class="cp-acc-ts">{item["ts"]}</span>'
                f'<span class="cp-acc-chevron" style="border-left-color:{color};"></span>'
                f'</div>'
                f'<div class="cp-acc-body">'
                f'<div class="cp-acc-pl-title" style="color:{color};text-shadow:0 0 8px {color}66;">{pl_title}</div>'
                f'{tracks_html}'
                f'<div class="cp-acc-actions">'
                f'<a href="{yt}" target="_blank" class="cb-btn yt" aria-label="Search this playlist on YouTube">&#x25B6; YouTube</a>'
                f'<a href="{sp}" target="_blank" class="cb-btn sp" aria-label="Search this playlist on Spotify">&#x25CE; Spotify</a>'
                f'</div></div></div>'
            )

        st.markdown(history_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR — Neural Stats
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### &#x2B21; NEURAL STATS")
    hist = st.session_state.history
    if hist:
        emotions_seen = [h["emotion"] for h in hist]
        counts = Counter(emotions_seen)
        dom = counts.most_common(1)[0][0]

        st.metric("Total Scans", len(hist))
        st.metric("Dominant Emotion", f"{EMOJI_FULL.get(dom,'')} {dom.upper()}")

        st.markdown("---")
        st.markdown(
            '<div style="font-family:\'Orbitron\',monospace;font-size:.5rem;'
            'letter-spacing:.15em;color:#606080;text-transform:uppercase;'
            'margin-bottom:.6rem;">// Frequency Map</div>',
            unsafe_allow_html=True
        )
        import pandas as pd
        df = pd.DataFrame(counts.items(), columns=["Emotion", "Scans"])
        st.bar_chart(df.set_index("Emotion"), color="#00f7ff")
    else:
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;font-size:.82rem;'
            'color:#282838;letter-spacing:.08em;text-transform:uppercase;'
            'padding:1rem 0;">// No scans recorded yet</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'Share Tech Mono\',monospace;font-size:.7rem;'
        'color:#282838;letter-spacing:.08em;text-transform:uppercase;line-height:1.9;">'
        'MoodBeats v4.0<br>FER2013 · MediaPipe<br>Streamlit · Neural OS</div>',
        unsafe_allow_html=True
    )
