"""Global CSS for the Streamlit dashboard."""
import streamlit as st


def inject_custom_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 10%, #0f172a 0%, #070b14 60%, #050811 100%);
        color: #f1f5f9;
    }

    /* Hero Banner with SVG Grid Pattern */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(12, 18, 34, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), 0 0 30px rgba(14, 165, 233, 0.08);
    }

    .hero-top-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 10px;
    }

    .top-header-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(14, 165, 233, 0.14);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .live-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse-green 2s infinite;
    }

    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    .page-title {
        font-size: 2.05rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 55%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        line-height: 1.2;
    }
    
    .page-subtitle {
        color: #94a3b8;
        font-size: 0.94rem;
        line-height: 1.5;
        margin: 0;
    }

    /* KPI Cards Grid */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }
    
    .kpi-card {
        background: linear-gradient(180deg, rgba(19, 29, 54, 0.82) 0%, rgba(11, 18, 35, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 2px solid rgba(56, 189, 248, 0.45);
        border-radius: 14px;
        padding: 16px 18px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(12px);
        transition: all 0.22s ease;
        box-shadow: 0 10px 25px -10px rgba(0, 0, 0, 0.5);
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 14px 30px -10px rgba(14, 165, 233, 0.2);
    }

    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    
    .kpi-title {
        font-size: 0.73rem;
        color: #94a3b8;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .kpi-icon {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        font-size: 0.95rem;
    }
    
    .kpi-value {
        font-size: 1.62rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    
    .kpi-badge {
        display: inline-block;
        margin-top: 8px;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
    }
    
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .badge-blue  { background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.35); }
    .badge-red   { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); }
    .badge-gold  { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }

    /* Section Containers */
    .glass-panel {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }

    /* Streamlit Components Overrides */
    div[data-testid="stSidebar"] {
        background-color: #090e1a;
        border-right: 1px solid rgba(56, 189, 248, 0.12);
    }
    
    div[data-testid="stMetric"] {
        background: rgba(18, 27, 49, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 14px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.75);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 9px 16px;
        font-size: 0.88rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.25) 0%, rgba(30, 41, 59, 0.95) 100%) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
    }

    .feature-box {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.75) 0%, rgba(10, 15, 29, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #38bdf8;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .kpi-card {
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .kpi-card:nth-child(1) { animation-delay: 0.05s; }
    .kpi-card:nth-child(2) { animation-delay: 0.10s; }
    .kpi-card:nth-child(3) { animation-delay: 0.15s; }
    .kpi-card:nth-child(4) { animation-delay: 0.20s; }
    .kpi-card:nth-child(5) { animation-delay: 0.25s; }
    .kpi-card:nth-child(6) { animation-delay: 0.30s; }

    /* Live Operations Ticker Tape */
    .ticker-wrap {
        width: 100%;
        overflow: hidden;
        background: rgba(11, 18, 35, 0.88);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 10px;
        padding: 8px 0;
        margin-bottom: 18px;
        white-space: nowrap;
        position: relative;
    }

    .ticker-move {
        display: inline-block;
        white-space: nowrap;
        padding-right: 100%;
        animation: ticker-scroll 34s linear infinite;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #cbd5e1;
    }

    .ticker-move:hover {
        animation-play-state: paused;
    }

    @keyframes ticker-scroll {
        0% { transform: translate3d(0, 0, 0); }
        100% { transform: translate3d(-50%, 0, 0); }
    }
    </style>
    """, unsafe_allow_html=True)
