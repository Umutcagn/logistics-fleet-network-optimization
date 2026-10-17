"""Small telemetry charts shown next to the map."""
import plotly.graph_objects as go


def build_fleet_donut_chart(assignments, total_spot):
    """
    Creates a sleek Donut chart showing vehicle mix across Tır, Kamyon, Hafif Kamyon, Kamyonet, and Spot.
    """
    counts = {"Tır (3000 Desi)": 0, "Kamyon (1500 Desi)": 0, "Hafif Kamyon (800)": 0, "Kamyonet (400)": 0, "Spot Kiralık": total_spot}
    label_map = {
        "tir": "Tır (3000 Desi)",
        "kamyon": "Kamyon (1500 Desi)",
        "hafif_kamyon": "Hafif Kamyon (800)",
        "kamyonet": "Kamyonet (400)"
    }
    for details in assignments.values():
        for v_key, cnt in details["fleet_vehicles"].items():
            counts[label_map.get(v_key, v_key)] = counts.get(label_map.get(v_key, v_key), 0) + cnt

    labels = [k for k, v in counts.items() if v > 0]
    values = [v for k, v in counts.items() if v > 0]
    colors = ["#38bdf8", "#10b981", "#818cf8", "#fbbf24", "#ef4444"][:len(labels)]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.64,
        marker=dict(colors=colors, line=dict(color="#0b1120", width=2)),
        textinfo="percent",
        textfont=dict(size=11, color="#ffffff"),
        hoverinfo="label+value+percent"
    )])
    fig.update_layout(
        title=dict(text="Aktif Filo Kompozisyonu", font=dict(size=13, color="#cbd5e1"), x=0.5, xanchor="center"),
        margin=dict(l=10, r=10, t=35, b=10),
        height=225,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=10, color="#94a3b8")),
        annotations=[dict(text=f"<b>{sum(values)}</b><br><span style='font-size:10px;color:#94a3b8'>ARAÇ</span>", x=0.5, y=0.5, font_size=16, showarrow=False, font=dict(color="#f8fafc"))]
    )
    return fig


def build_k5_gauge_chart(avg_fill):
    """
    Creates an executive gauge chart for K5 Average Fleet Utilization.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_fill,
        number={"suffix": "%", "font": {"size": 26, "color": "#f8fafc", "family": "JetBrains Mono"}},
        delta={"reference": 80.0, "increasing": {"color": "#10b981"}, "decreasing": {"color": "#f59e0b"}},
        title={"text": "K5 Filo Doluluk Performansı", "font": {"size": 13, "color": "#cbd5e1"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
            "bar": {"color": "#38bdf8", "thickness": 0.35},
            "bgcolor": "rgba(15, 23, 42, 0.8)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.08)",
            "steps": [
                {"range": [0, 65], "color": "rgba(239, 68, 68, 0.18)"},
                {"range": [65, 80], "color": "rgba(245, 158, 11, 0.18)"},
                {"range": [80, 100], "color": "rgba(16, 185, 129, 0.22)"}
            ],
            "threshold": {
                "line": {"color": "#10b981", "width": 3},
                "thickness": 0.75,
                "value": 85.0
            }
        }
    ))
    fig.update_layout(
        margin=dict(l=18, r=18, t=38, b=10),
        height=215,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc")
    )
    return fig
