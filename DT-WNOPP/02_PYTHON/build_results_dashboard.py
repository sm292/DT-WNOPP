"""Build the complete DT-WNOPP results dashboard and all individual figures."""
from __future__ import annotations
import argparse
import base64
from datetime import datetime
from html import escape
from pathlib import Path
from visualization_common import output_dir, project_root
from plot_validation_results import generate_validation
from plot_system_results import generate_system


def image_block(path: Path, caption: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>'


def make_dashboard(root: Path, out: Path) -> Path:
    validation_figures, validation_notes = generate_validation(root, out)
    system_figures, scenario_notes, proxy_notes = generate_system(root, out)
    notes = [*validation_notes, *scenario_notes, *proxy_notes]
    (out / "results_interpretation.txt").write_text("\n\n".join(notes) + "\n", encoding="utf-8")
    figure_captions = {
        "validation_throughput.png": "Throughput error and R², averaged across grouped holdout splits.",
        "validation_jitter.png": "Jitter error and R², averaged across grouped holdout splits.",
        "validation_packet_loss.png": "Packet-loss error and R², averaged across grouped holdout splits.",
        "measured_resource_scenarios.png": "Observed configuration-level outcomes by resource fraction; configurations are not paired.",
        "matched_simulink_proxy_results.png": "Baseline and candidate outputs from the Simulink proxy model.",
    }
    figures = validation_figures + system_figures
    validation_html = "\n".join(image_block(p, figure_captions.get(p.name, p.stem)) for p in validation_figures)
    system_html = "\n".join(image_block(p, figure_captions.get(p.name, p.stem)) for p in system_figures)
    validation_list = "\n".join(f"<li>{escape(note)}</li>" for note in validation_notes)
    scenario_list = "\n".join(f"<li>{escape(note)}</li>" for note in scenario_notes)
    proxy_list = "\n".join(f"<li>{escape(note)}</li>" for note in proxy_notes)
    manifest_path = root / "04_RESULTS" / "validation_run_manifest.json"
    created = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    manifest_note = "The validation manifest was not found." if not manifest_path.exists() else "Grouped holdout validation manifest found; inspect validation_run_manifest.json for seed, features, target definitions, source-data hash, and software versions."
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DT-WNOPP Results Dashboard</title>
<style>
:root {{ color-scheme: light; font-family: Segoe UI, Arial, sans-serif; color: #203040; background: #f4f7fa; }}
body {{ margin: 0; }} main {{ max-width: 1220px; margin: 0 auto; padding: 28px 22px 50px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; }} h2 {{ margin: 34px 0 8px; font-size: 22px; }} p, li {{ line-height: 1.55; }}
.subtle {{ color: #526273; }} .notice {{ padding: 16px 18px; background: #fff3df; border-left: 5px solid #dc8d2e; margin: 20px 0; line-height: 1.55; }}
section {{ margin-top: 22px; }} figure {{ margin: 22px 0 34px; }} figure img {{ display: block; width: 100%; height: auto; background: white; }} figcaption {{ color: #526273; font-size: 14px; margin-top: 8px; }}
ul {{ padding-left: 22px; }} footer {{ border-top: 1px solid #d8e0e8; margin-top: 40px; padding-top: 14px; color: #526273; font-size: 13px; }} code {{ overflow-wrap: anywhere; }}
@media (max-width: 600px) {{ main {{ padding: 18px 12px 36px; }} h1 {{ font-size: 25px; }} h2 {{ font-size: 19px; }} }}
</style></head><body><main>
<h1>DT-WNOPP Results Dashboard</h1>
<p class="subtle">Generated {escape(created)} from the project’s saved validation, scenario-summary, and Simulink-proxy CSV outputs.</p>
<div class="notice"><strong>Evidence boundary:</strong> throughput prediction is promising, but jitter and packet-loss prediction are weak. The resource-level measurements do not compare the same configurations. Simulink candidate results are outputs of the current proxy model, not independent network measurements; they do not establish a real-world optimum.</div>
<h2>Validation results</h2><ul>{validation_list}</ul>
{validation_html}
<h2>Observed resource scenarios</h2><ul>{scenario_list}</ul>
{image_block(next(p for p in system_figures if p.name == 'measured_resource_scenarios.png'), figure_captions['measured_resource_scenarios.png'])}
<h2>Matched Simulink proxy runs</h2><ul>{proxy_list}</ul>
{image_block(next(p for p in system_figures if p.name == 'matched_simulink_proxy_results.png'), figure_captions['matched_simulink_proxy_results.png'])}
<footer><p>{escape(manifest_note)}</p><p>Visuals use saved results; they do not retrain models or rerun MATLAB. Refresh them by rerunning the Python visualization scripts after the source result CSVs change.</p><p>Project results: <code>{escape(str(root / '04_RESULTS'))}</code></p></footer>
</main></body></html>'''
    dashboard = out / "dt_wnopp_results_dashboard.html"
    dashboard.write_text(html, encoding="utf-8")
    return dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", help="DT-WNOPP project folder; defaults to the known project path.")
    parser.add_argument("--output-dir", help="Folder for individual diagrams, interpretation, and dashboard.")
    args = parser.parse_args()
    root = project_root(__file__, args.project_root)
    out = output_dir(root, args.output_dir)
    dashboard = make_dashboard(root, out)
    print(f"Dashboard: {dashboard}")
    print(f"Figures and interpretation: {out}")

if __name__ == "__main__":
    main()
