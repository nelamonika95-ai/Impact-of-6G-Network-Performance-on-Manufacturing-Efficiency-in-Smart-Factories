"""Chart theme for the 6G smart-factory dashboard.

Palette values are taken from the validated reference instance and checked with
the data-viz validator against the light chart surface (#fcfcfb):

  categorical 3 slots  "#2a78d6,#eb6834,#1baf7a"  --mode light --pairs all
      -> lightness band PASS, chroma floor PASS,
         worst all-pairs CVD dE 9.2 (deutan), normal-vision dE 24.0,
         WARN aqua 2.74:1 vs surface -> relief rule satisfied: every chart in
         this app ships direct labels and a table view.

  ordinal 3-step blue  "#86b6ef,#2a78d6,#104281"  --mode light --ordinal
      -> monotone L PASS, adjacent dL PASS, light-end 2.06:1 PASS,
         single hue (3 deg spread) PASS.

The app pins Streamlit's light theme in .streamlit/config.toml, so the surface
these were validated against is the surface that actually renders.
"""

# ---------------------------------------------------------------- surfaces/ink
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# ------------------------------------------------------------------ categorical
# Fixed slot order - never cycled, never reassigned by rank.
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
MODE_COLORS = {"Active": CAT[0], "Idle": CAT[1], "Maintenance": CAT[2]}

# ------------------------------------------------ ordinal ramp (Low->High)
# Efficiency_Status and Network_Quality are ordered categories, so they get a
# single-hue ordinal ramp, not categorical hues.
ORDINAL = {"Low": "#86b6ef", "Medium": "#2a78d6", "High": "#104281"}
EFF_ORDER = ["Low", "Medium", "High"]
BAND_ORDER = ["Low", "Medium", "High"]

# ------------------------------------------------------------ sequential ramp
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]

# ---------------------------------------------------------------------- status
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

# ------------------------------------------------------------- emphasis / gray
EMPHASIS = "#2a78d6"
DEEMPHASIS = "#c3c2b7"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def layout(height=340, showlegend=True, **kw):
    """Recessive-chrome Plotly layout: hairline solid grid, no dashes, padding.

    autosize lets width follow the Streamlit column instead of Plotly's 700px
    default. Height stays fixed and is sized to include the x-axis label band,
    so a card never gets a nested vertical scrollbar.
    """
    base = dict(
        height=height,
        autosize=True,
        showlegend=showlegend,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_2),
        margin=dict(l=64, r=24, t=48, b=56),
        title=dict(font=dict(size=14, color=INK), x=0, xanchor="left"),
        hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor=SURFACE,
                        bordercolor=AXIS, font_color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11, color=INK_2), title_text=""),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=AXIS, linewidth=1,
                   ticks="outside", tickcolor=AXIS, ticklen=4,
                   tickfont=dict(size=11, color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                   linecolor=AXIS, linewidth=1,
                   tickfont=dict(size=11, color=MUTED)),
    )
    base.update(kw)
    return base


CSS = f"""
<style>
  .stApp {{ background: {PAGE}; }}
  .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
  html, body, [class*="css"] {{ font-family: {FONT}; }}

  .hero {{ font-size: 54px; line-height: 1.05; font-weight: 650;
           color: {INK}; letter-spacing: -0.02em; }}
  .hero-sub {{ font-size: 13px; color: {INK_2}; margin-top: 4px; }}

  .tile {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10);
           border-radius: 10px; padding: 14px 16px; height: 100%; }}
  .tile-label {{ font-size: 11px; text-transform: uppercase;
                 letter-spacing: 0.07em; color: {MUTED}; font-weight: 600; }}
  .tile-value {{ font-size: 30px; font-weight: 640; color: {INK};
                 line-height: 1.15; margin-top: 6px; }}
  .tile-note {{ font-size: 11.5px; color: {INK_2}; margin-top: 5px; }}

  .verdict {{ border-radius: 10px; padding: 14px 16px; margin: 6px 0 14px 0;
              font-size: 13.5px; line-height: 1.55; border: 1px solid; }}
  .v-null {{ background: rgba(208,59,59,0.06); border-color: rgba(208,59,59,0.35);
             color: {INK}; }}
  .v-real {{ background: rgba(12,163,12,0.06); border-color: rgba(12,163,12,0.35);
             color: {INK}; }}
  .v-info {{ background: rgba(42,120,214,0.06); border-color: rgba(42,120,214,0.30);
             color: {INK}; }}
  .verdict b {{ color: {INK}; }}

  .caption {{ font-size: 11.5px; color: {MUTED}; margin: -6px 0 12px 0; }}
  .swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px;
             margin-right:6px; vertical-align:middle; }}
  section[data-testid="stSidebar"] {{ background: {SURFACE}; }}
  div[data-testid="stMetricValue"] {{ font-size: 26px; }}
  thead tr th {{ font-size: 11.5px !important; }}
  tbody tr td {{ font-variant-numeric: tabular-nums; }}
</style>
"""


def tile(label, value, note=""):
    return (f'<div class="tile"><div class="tile-label">{label}</div>'
            f'<div class="tile-value">{value}</div>'
            f'<div class="tile-note">{note}</div></div>')


def verdict(kind, html):
    return f'<div class="verdict v-{kind}">{html}</div>'
