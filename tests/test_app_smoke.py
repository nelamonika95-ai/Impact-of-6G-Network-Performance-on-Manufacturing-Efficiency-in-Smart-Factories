"""Headless smoke tests for the dashboard.

The panels all compute confidence intervals, bin counts and equivalence tests on
whatever slice the filters produce, so the failure mode that matters is a slice
too small or too degenerate for those computations. These tests drive the real
app through Streamlit's own test harness and assert it never raises - across
every module, and on single-class, single-machine and single-hour slices.

Only the selected module renders, so every module must be selected explicitly;
a crash in an unselected module is invisible.

Run:  .venv/Scripts/python.exe -m pytest tests -q
"""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = ["Network performance overview", "Network vs efficiency",
           "Quality & error impact", "6G optimization insights",
           "Efficiency diagnostics", "Data & method"]

WIDGET_KEYS = {
    "module": ("radio", "f_module"),
    "time_window": ("radio", "f_window"),
    "hours": ("slider", "f_hours"),
    "bands": ("multiselect", "f_bands"),
    "effs": ("multiselect", "f_effs"),
    "modes": ("multiselect", "f_modes"),
    "machines": ("multiselect", "f_machines"),
}

TIMEOUT = 180


def run(**widgets):
    """Run app.py to completion with the given sidebar values.

    Widgets are addressed by key, not by index - the app also has in-panel
    radios (the heatmap metric selector), so positional lookup silently targets
    the wrong control.
    """
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, f"baseline run raised: {at.exception}"

    for name, value in widgets.items():
        kind, key = WIDGET_KEYS[name]
        getattr(at, kind)(key=key).set_value(value)
    at.run()
    return at


def body(at):
    return " ".join(m.value for m in at.markdown)


# --------------------------------------------------------------- every module
@pytest.mark.parametrize("module", MODULES)
def test_module_renders(module):
    at = run(module=module)
    assert not at.exception, f"{module} raised: {at.exception}"
    assert at.markdown, f"{module} produced no output"


def test_headline_findings_present():
    at = run()
    text = body(at)
    assert "Headline finding" in text
    at5 = run(module="Efficiency diagnostics")
    assert "99.998" in body(at5)          # recovered-rule accuracy
    at4 = run(module="6G optimization insights")
    assert "equivalent to zero" in body(at4)


# ------------------------------------------------- degenerate slices x module
@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("effs", [["Low"], ["High"]])
def test_single_efficiency_class(module, effs):
    """One class => chi-square, Kruskal-Wallis and eta^2 all go degenerate."""
    at = run(module=module, effs=effs)
    assert not at.exception, f"{module} / effs={effs} raised: {at.exception}"


@pytest.mark.parametrize("module", MODULES)
def test_tiny_slice(module):
    """~30 rows: the smallest realistic slice the filters can produce."""
    at = run(module=module, time_window="Last 7 days", effs=["High"],
             modes=["Maintenance"])
    assert not at.exception, f"{module} raised: {at.exception}"


@pytest.mark.parametrize("module", MODULES)
def test_single_machine_single_hour(module):
    """~1 row per day: bin counts collapse below every qcut threshold."""
    at = run(module=module, machines=[7], hours=(3, 3))
    assert not at.exception, f"{module} raised: {at.exception}"


@pytest.mark.parametrize("modes", [["Maintenance"], ["Idle"], ["Active"]])
def test_single_operation_mode(modes):
    at = run(modes=modes)
    assert not at.exception, f"modes={modes} raised: {at.exception}"


def test_single_band_single_class_single_mode():
    at = run(module="6G optimization insights", bands=["Low"], effs=["High"],
             modes=["Maintenance"], time_window="Last 7 days")
    assert not at.exception


def test_empty_multiselects_mean_no_filter():
    at = run(effs=[], bands=[], modes=[], machines=[])
    assert not at.exception
    assert "100,000" in body(at), "empty selections should not filter anything out"


@pytest.mark.parametrize("hours", [(0, 0), (23, 23), (0, 23), (12, 13)])
def test_hour_window_boundaries(hours):
    at = run(hours=hours)
    assert not at.exception, f"hours={hours} raised: {at.exception}"
