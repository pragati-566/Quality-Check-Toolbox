"""
live_html.py — Generates a live, standalone HTML QC dashboard.
Uses base64-encoded images to keep it a single self-contained file.
"""

import base64
import io
import json
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qc_toolbox.visualize import plot_cbf_slice


def _get_base64_plot(cbf_map, title: str) -> str:
    """Generate a CBF slice plot and return it as a base64 PNG string."""
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#1e212b")
    plot_cbf_slice(cbf_map, title, ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_base64_mask_overlay(cbf_map, gm_mask, wm_mask, title: str) -> str:
    """Plot CBF slice with GM (red) and WM (blue) mask contours overlaid."""
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#1e212b")
    plot_cbf_slice(cbf_map, title, ax)
    
    from qc_toolbox.visualize import _mid_slice
    
    # Overlay contours if masks are not empty
    if gm_mask.any():
        ax.contour(_mid_slice(gm_mask).T, levels=[0.5], colors=["#ff4757"], linewidths=0.8, alpha=0.8)
    if wm_mask.any():
        ax.contour(_mid_slice(wm_mask).T, levels=[0.5], colors=["#3742fa"], linewidths=0.8, alpha=0.8)
    
    # Custom legend for contours
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color="#ff4757", lw=2),
                    Line2D([0], [0], color="#3742fa", lw=2)]
    ax.legend(custom_lines, ['GM Bound', 'WM Bound'], loc="upper right", 
              fontsize=7, facecolor="#1e212b", edgecolor="#333", labelcolor="white")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_base64_histogram(cbf_map, gm_mask, wm_mask, csf_mask) -> str:
    """Plot CBF distributions for GM/WM/CSF."""
    fig, ax = plt.subplots(figsize=(4, 3), facecolor="#1e212b")
    ax.set_facecolor("#1e212b")
    
    if gm_mask.any(): ax.hist(cbf_map[gm_mask], bins=50, color="#ff4757", alpha=0.6, label="GM")
    if wm_mask.any(): ax.hist(cbf_map[wm_mask], bins=50, color="#3742fa", alpha=0.6, label="WM")
    if csf_mask.any(): ax.hist(cbf_map[csf_mask], bins=50, color="#2ed573", alpha=0.6, label="CSF")
    
    ax.set_title("CBF Distribution", color="white", fontsize=10)
    ax.tick_params(colors="#a4b0be", labelsize=8)
    for spine in ax.spines.values(): spine.set_color("#333")
    ax.legend(loc="upper right", fontsize=8, facecolor="#1e212b", edgecolor="#333", labelcolor="white")
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_base64_timeseries(timeseries: list[float], context: list[str]) -> str:
    """Plot Control/Label alternate timeseries."""
    fig, ax = plt.subplots(figsize=(4, 2), facecolor="#1e212b")
    ax.set_facecolor("#1e212b")
    
    if timeseries and context:
        x = np.arange(len(timeseries))
        ax.plot(x, timeseries, color="#a4b0be", linewidth=1.5, zorder=1)
        
        # Overlay scatter points colored by context
        c_idx = [i for i, ctx in enumerate(context) if ctx.lower() == "control"]
        l_idx = [i for i, ctx in enumerate(context) if ctx.lower() == "label"]
        
        if c_idx: ax.scatter(c_idx, [timeseries[i] for i in c_idx], color="#2ed573", s=20, zorder=2, label="Control")
        if l_idx: ax.scatter(l_idx, [timeseries[i] for i in l_idx], color="#ff4757", s=20, zorder=2, label="Label")
        
    ax.set_title("Mean Signal Timecourse", color="white", fontsize=10)
    ax.tick_params(colors="#a4b0be", labelsize=8)
    for spine in ax.spines.values(): spine.set_color("#333")
    ax.legend(loc="upper right", fontsize=7, facecolor="#1e212b", edgecolor="#333", labelcolor="white")
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_live_html(results: list[dict], total_subjects: int, output_path: str):
    """
    Generate an aesthetic, dark-mode glassmorphic HTML report.
    `results` must contain the dict from _qc_subject PLUS the 'cbf_map' for plotting.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress = int((len(results) / total_subjects) * 100) if total_subjects else 100
    
    cards_html = ""
    for r in reversed(results):
        # Determine status color
        color = "#ff4757" if r["flagged"] else "#2ed573"
        status_text = "FLAGGED" if r["flagged"] else "PASS"
        flags = r["flags"] if r["flagged"] else "All metrics within threshold"
        
        # Generate inline images
        title = f"{r['subject_id']} (QEI: {r['qei']:.2f})"
        b64_slice = _get_base64_plot(r["cbf_map"], title)
        b64_mask  = _get_base64_mask_overlay(r["cbf_map"], r.get("gm_mask", np.array([])), r.get("wm_mask", np.array([])), "Tissue Masks")
        b64_hist  = _get_base64_histogram(r["cbf_map"], r.get("gm_mask", np.array([])), r.get("wm_mask", np.array([])), r.get("csf_mask", np.array([])))
        b64_time  = _get_base64_timeseries(r.get("raw_timeseries", []), r.get("asl_context", []))
        
        card = f"""
        <div class="card">
            <div class="plots-grid">
                <div class="main-image-row">
                    <img src="data:image/png;base64,{b64_slice}" alt="CBF Map">
                    <img src="data:image/png;base64,{b64_mask}" alt="Mask Overlay">
                </div>
                <div class="sub-plots">
                    <img src="data:image/png;base64,{b64_hist}" alt="CBF Histogram">
                    <img src="data:image/png;base64,{b64_time}" alt="Timeseries Plot">
                </div>
            </div>
            <div class="card-content">
                <h3>{r['subject_id']}</h3>
                <div class="badge" style="background-color: {color}22; color: {color}; border: 1px solid {color}55;">
                    {status_text}
                </div>
                <div class="metrics">
                    <p><strong>QEI:</strong> {r['qei']:.3f}</p>
                    <p><strong>PSS:</strong> {r['pss']:.3f}</p>
                    <p><strong>DI:</strong> {r['di']:.2f}</p>
                    <p><strong>Neg GM:</strong> {r['n_gm']:.1%}</p>
                </div>
                <p class="flags">{flags}</p>
            </div>
        </div>
        """
        cards_html += card

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QC Live Dashboard</title>
    <meta http-equiv="refresh" content="10"> <!-- Auto refresh every 10s -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0f111a;
            --surface: rgba(30, 33, 43, 0.7);
            --border: rgba(255, 255, 255, 0.1);
            --text-main: #f1f2f6;
            --text-muted: #a4b0be;
            --accent: #3742fa;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 2rem;
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(55, 66, 250, 0.08), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(46, 213, 115, 0.08), transparent 25%);
        }}
        .header {{
            max-width: 1200px;
            margin: 0 auto 2rem;
            text-align: center;
        }}
        .header h1 {{ margin: 0; font-weight: 800; font-size: 2.5rem; letter-spacing: -1px; }}
        .header p {{ color: var(--text-muted); margin-top: 0.5rem; }}
        
        .progress-container {{
            max-width: 600px;
            margin: 2rem auto;
            background: rgba(0,0,0,0.3);
            border-radius: 20px;
            height: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        .progress-bar {{
            height: 100%;
            background: linear-gradient(90deg, #3742fa, #5352ed);
            width: {progress}%;
            transition: width 0.5s ease;
        }}
        .progress-text {{
            text-align: center;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-muted);
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 1.5rem;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .card {{
            background: var(--surface);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s;
        }}
        .card:hover {{ transform: translateY(-4px); }}
        
        .plots-grid {{
            display: flex;
            flex-direction: column;
            border-bottom: 1px solid var(--border);
        }}
        .main-image-row {{
            display: flex;
            flex-direction: row;
        }}
        .main-image-row img {{
            width: 50%;
            height: auto;
            object-fit: contain;
        }}
        .sub-plots {{
            display: flex;
            flex-direction: row;
            background: #1e212b;
            border-top: 1px solid var(--border);
        }}
        .sub-plots img {{ width: 50%; height: auto; object-fit: contain; }}
        
        .card-content {{ padding: 1.5rem; }}
        .card-content h3 {{ margin: 0 0 0.5rem 0; font-size: 1.2rem; }}
        
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            margin-bottom: 1rem;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }}
        .metrics p {{ margin: 0; }}
        .metrics strong {{ color: var(--text-main); }}
        
        .flags {{
            font-size: 0.8rem;
            color: #ff4757;
            margin: 0;
            padding: 0.75rem;
            background: rgba(255, 71, 87, 0.1);
            border-radius: 8px;
            border: 1px dashed rgba(255, 71, 87, 0.3);
        }}
        .loader {{
            display: inline-block;
            width: 12px; height: 12px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
            margin-right: 8px;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Quality Check Toolbox</h1>
        <p>Live Pipeline Execution Dashboard</p>
        <div class="progress-text">
            {f'<div class="loader"></div> Processing Subject {len(results)+1} of {total_subjects}...' if progress < 100 else '✅ Pipeline Complete'}
            <br><small style="opacity:0.5">Last updated: {now}</small>
        </div>
        <div class="progress-container">
            <div class="progress-bar"></div>
        </div>
    </div>
    <div class="grid">
        {cards_html}
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
