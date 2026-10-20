"""HTML rendering for the benchmark overview."""

from __future__ import annotations

from typing import Any

from .common import escape


def render_index_html(payload: dict[str, Any]) -> str:
    """Render the cross-machine index as a self-contained sortable table."""
    columns = (
        ("model_id", "Model"),
        ("precision", "Precision"),
        ("provider", "Provider"),
        ("mode", "Mode"),
        ("gpu_name", "GPU"),
        ("os_name", "OS"),
        ("machine_label", "Machine"),
        ("images_per_second", "Images/s"),
        ("ms_per_image", "ms/image"),
        ("cv", "CV"),
    )
    header = "".join(f'<th scope="col">{escape(label)}</th>' for _key, label in columns)
    body: list[str] = []
    for row in payload.get("rows") or []:
        cells = []
        for key, _label in columns:
            value = row.get(key)
            if key in {"images_per_second", "ms_per_image"} and value is not None:
                value = f"{float(value):.2f}"
            elif key == "cv" and value is not None:
                value = f"{float(value) * 100.0:.1f}%"
            cells.append(f"<td>{escape(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Benchmark overview</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 2rem auto; max-width: 82rem; padding: 0 1rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
  th, td {{ border-bottom: 1px solid rgba(128,128,128,.35); padding: .35rem .55rem; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ position: sticky; top: 0; background: Canvas; }}
  tbody tr:hover {{ background: color-mix(in srgb, CanvasText 6%, Canvas); }}
  .muted {{ opacity: .7; }}
</style>
</head>
<body>
<h1>Benchmark overview</h1>
<p class="muted">{payload.get('row_count', 0)} rows · generated {escape(payload.get('generated_at', ''))}</p>
<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>
</body>
</html>
"""


__all__ = ["render_index_html"]
