"""
MotherDuck-style Gradio app for Nabat-AI.

Modes:
    - Audio Upload
    - Record Audio
    - Text Input
    - Search Corpus
    - Poetry Map
"""

# ruff: noqa: F403, F405, E402
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOT_CACHE_DIR = PROJECT_ROOT / "outputs" / ".cache"
BOOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(BOOT_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(BOOT_CACHE_DIR / "matplotlib"))

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402
from loguru import logger  # noqa: E402

matplotlib.use("Agg")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MAP_CACHE_PATH = REPORT_DIR / "poetry_map_points.json"
PLOT_BG = "#f4efea"
CARD_BG = "#fbf8f5"
BORDER = "#d7d0ca"
TEXT = "#383838"
MUTED = "#7f746a"
ACCENT = "#706458"

CORPUS_PATHS = [
    PROJECT_ROOT / "data/processed/master_dataset.jsonl",
    PROJECT_ROOT / "data/processed/train.jsonl",
    PROJECT_ROOT / "data/processed/val.jsonl",
    PROJECT_ROOT / "data/processed/test.jsonl",
]

_GENRE_CACHE: dict[str, tuple[Any, Any]] = {}
_EMOTION_CACHE: dict[str, tuple[Any, Any]] = {}
_AROUSAL_CACHE: dict[str, tuple[Any, Any]] = {}
_CNN_CACHE: dict[str, Any] = {}
_RETRIEVER_CACHE: dict[str, Any] = {}
# Populated once in build_ui() so helper functions can embed icons without needing icon_map passed as args
_UI_ICON_MAP: dict[str, str] = {}


from src.ui.app_helpers import *
from src.ui.app_helpers import _load_imagery_corpus


def build_ui(default_device: str = "cpu") -> gr.Blocks:
    global _UI_ICON_MAP
    initial_map, initial_points, _ = build_map_figure("Genre", "")
    icon_map = discover_icons()
    _UI_ICON_MAP.update(icon_map)
    with gr.Blocks(title="Nabat-AI", fill_width=True) as ui:
        gr.HTML(f"<style>{APP_CSS}</style>")
        gr.HTML(
            """
            <div class="summary-card hero-card" style="background:#3a7d44;color:#fff;border-color:#3a7d44;">
              <div class="eyebrow" style="color:#fff;">Nabat-AI Demo</div>
              <div class="metric-value" style="color:#fff;">A local, offline Nabati poetry demo for analysis and corpus search.</div>
              <div class="small" style="color:#fff;">The app is built for an English-speaking reader: concise summaries, confidence badges, and corrected corpus text when available.</div>
            </div>
            """
        )
        setup_notice = offline_setup_notice_html()
        if setup_notice:
            gr.HTML(setup_notice)

        with gr.Column(visible=True) as start_screen:
            gr.HTML(
                """
                <div class="start-shell">
                  <div class="mode-copy">
                    <div class="eyebrow">Start Here</div>
                    <h2>Choose one path.</h2>
                    <p>Pick analysis when you want an interpretable reading of the poem. Pick search when you already have audio and mainly want the closest corpus match.</p>
                  </div>
                </div>
                """,
                elem_id="start-copy",
            )
            with gr.Row(equal_height=True):
                with gr.Column(elem_classes=["mode-card"], elem_id="analyze-mode-card"):
                    gr.HTML(
                        render_start_icon(
                            icon_map.get("analyze"), "analyze", "~", "Analyze mode icon"
                        )
                    )
                    gr.HTML(
                        """
                        <div class="mode-tag">Analysis</div>
                        <h3>Analyze a poem / clip</h3>
                        <p>Upload audio, record audio, or paste Arabic poem text. This path explains genre, poem emotion, delivery, and similar poems in English-friendly language.</p>
                        """
                    )
                    analyze_start = gr.Button(
                        "Open Analyze Mode",
                        elem_classes=["mode-cta", "mode-cta-analyze"],
                    )
                with gr.Column(elem_classes=["mode-card"], elem_id="search-mode-card"):
                    gr.HTML(
                        render_start_icon(
                            icon_map.get("search"), "search", "?", "Search mode icon"
                        )
                    )
                    gr.HTML(
                        """
                        <div class="mode-tag">Retrieval</div>
                        <h3>Search for a poem in the corpus</h3>
                        <p>Upload or record audio to find the most similar poem in the corpus. This path emphasizes ranking, similarity, and candidate review instead of classification.</p>
                        """
                    )
                    search_start = gr.Button(
                        "Open Search Mode",
                        elem_classes=["mode-cta", "mode-cta-search"],
                    )

        with gr.Column(visible=False) as analyze_group:
            analyze_back = gr.Button("Back to Start Screen", elem_classes=["back-cta"])
            gr.HTML(
                f"""
                <div class="card">
                  <div class="section-title">{render_inline_icon(icon_map.get("analyze"), "Dallah – Analyze mode")}Analyze a poem / clip</div>
                  <div class="small">Use this when you want a full reading of a poem: genre, poem-level emotions, delivery, and similar poems.</div>
                </div>
                """
            )
            with gr.Row():
                analyze_top_k = gr.Slider(
                    1, 10, value=5, step=1, label="Retrieval depth", visible=False
                )
                analyze_imagery = gr.Textbox(
                    label="Imagery tag filter",
                    placeholder="heart, desert, night",
                    value="",
                    visible=False,
                )
                analyze_device = gr.Radio(
                    choices=["cpu", "mps", "cuda"],
                    value=default_device,
                    label="Compute device",
                )

            with gr.Tabs():
                with gr.Tab("Upload audio"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            upload_audio = gr.Audio(
                                type="filepath", sources=["upload"], label="Upload clip"
                            )
                            upload_lora = gr.Checkbox(
                                label="Use LoRA (experimental)", value=False
                            )
                            upload_btn = gr.Button("Analyze upload", variant="primary")
                        upload_outputs = build_result_panel("Upload", include_retrieval=False, minimal=True)
                    _upload_evt = upload_btn.click(
                        fn=analyse_audio,
                        inputs=[
                            upload_audio,
                            analyze_top_k,
                            analyze_imagery,
                            analyze_device,
                            upload_lora,
                        ],
                        outputs=list(upload_outputs),
                    )
                    upload_dl_btn = gr.DownloadButton(
                        "📥 Download poem JSON", size="sm", visible=True
                    )
                    _upload_evt.then(
                        fn=download_poem_json,
                        inputs=[upload_outputs[8]],
                        outputs=[upload_dl_btn],
                    )

                with gr.Tab("Record audio"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            record_audio = gr.Audio(
                                type="filepath",
                                sources=["microphone"],
                                label="Record clip",
                            )
                            record_lora = gr.Checkbox(
                                label="Use LoRA (experimental)", value=False
                            )
                            record_btn = gr.Button(
                                "Analyze recording", variant="primary"
                            )
                        record_outputs = build_result_panel("Record", include_retrieval=False, minimal=True)
                    _record_evt = record_btn.click(
                        fn=analyse_audio,
                        inputs=[
                            record_audio,
                            analyze_top_k,
                            analyze_imagery,
                            analyze_device,
                            record_lora,
                        ],
                        outputs=list(record_outputs),
                    )
                    record_dl_btn = gr.DownloadButton(
                        "📥 Download poem JSON", size="sm", visible=True
                    )
                    _record_evt.then(
                        fn=download_poem_json,
                        inputs=[record_outputs[8]],
                        outputs=[record_dl_btn],
                    )

                with gr.Tab("Paste Arabic poem text"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            text_in = gr.Textbox(
                                label="Arabic poem text",
                                lines=10,
                                placeholder="Paste a verse, a full poem, or a known poem_id.",
                            )
                            text_btn = gr.Button("Analyze text", variant="primary")
                        text_outputs = build_result_panel("Text", include_retrieval=False, minimal=True)
                    _text_evt = text_btn.click(
                        fn=analyse_text_mode,
                        inputs=[
                            text_in,
                            analyze_top_k,
                            analyze_imagery,
                            analyze_device,
                        ],
                        outputs=list(text_outputs),
                    )
                    text_dl_btn = gr.DownloadButton(
                        "📥 Download poem JSON", size="sm", visible=True
                    )
                    _text_evt.then(
                        fn=download_poem_json,
                        inputs=[text_outputs[8]],
                        outputs=[text_dl_btn],
                    )

            with gr.Accordion("Poetry Map", open=True):
                gr.HTML(
                    f"""
                    <div class="card">
                      <div class="section-title">{render_inline_icon(icon_map.get("map"), "Dihn Oud – Poetry Map")}What am I looking at?</div>
                      <div class="small">Each point is a poem-level embedding projected with t-SNE. Use the color switch to inspect how genre, dominant emotion, poet, or arousal proxy organize the corpus. Search narrows the visible set, then use the selector to open a poem card.</div>
                    </div>
                    """
                )
                with gr.Row():
                    map_color = gr.Dropdown(
                        choices=["Genre", "Emotion", "Poet", "Arousal"],
                        value="Genre",
                        label="Color by",
                    )
                    map_search = gr.Textbox(
                        label="Search poem_id / poet",
                        placeholder="e.g. poem0070_ku or Saad",
                    )
                    map_refresh = gr.Button("Refresh map")
                map_plot = gr.Plot(label="Interactive Poetry Map", value=initial_map)
                map_state = gr.State(initial_points)
                map_choice = gr.State(
                    [p["poem_id"] for p in initial_points if p.get("poem_id")]
                )
                map_refresh.click(
                    fn=build_map_figure,
                    inputs=[map_color, map_search],
                    outputs=[map_plot, map_state, map_choice],
                )

                # After each analysis finishes, populate map search with the poem_id
                # and immediately refresh the map so it highlights the analyzed poem.
                for _evt, _outputs in [
                    (_upload_evt, upload_outputs),
                    (_record_evt, record_outputs),
                    (_text_evt, text_outputs),
                ]:
                    (
                        _evt.then(
                            fn=update_map_search_for_corpus,
                            inputs=[_outputs[8]],
                            outputs=[map_search],
                        ).then(
                            fn=build_map_figure,
                            inputs=[map_color, map_search],
                            outputs=[map_plot, map_state, map_choice],
                        )
                    )

        with gr.Column(visible=False) as search_group:
            search_back = gr.Button("Back to Start Screen", elem_classes=["back-cta"])
            gr.HTML(
                f"""
                <div class="card">
                  <div class="section-title">{render_inline_icon(icon_map.get("search"), "Finjan – Search mode")}Search for a poem (audio) in the corpus</div>
                  <div class="small">Use this when your main goal is retrieval: find which corpus poem is closest to the input audio. Similarity score is the main confidence signal here.</div>
                </div>
                """
            )
            with gr.Accordion("🌸 Imagery Explorer — explore the corpus", open=True):
                gr.HTML(
                    f"""
                    <div class="card">
                      <div class="section-title">{render_inline_icon(icon_map.get("imagery"), "Dates – Imagery Explorer")}Imagery Explorer</div>
                      <div class="small">Browse the corpus before you search.
                      Discover which imagery motifs and emotional registers define each poet and genre.
                      Then use the upload / record tabs below to find similar poems.</div>
                    </div>
                    """
                )
                with gr.Tabs():
                    with gr.Tab("🧬 Poet Fingerprint"):
                        gr.HTML(
                            """<div class="small" style="margin-bottom:0.75rem;">
                            Select a poet to see their 12 most <em>distinctive</em> imagery tags —
                            scored by TF-IDF so common corpus-wide tags (e.g. "heart") are down-weighted
                            in favour of tags that uniquely characterise this poet's voice.
                            Clip count (×N) shows how often each tag appears in their corpus.
                            </div>"""
                        )
                        _records, _poets, _genres = _load_imagery_corpus()
                        with gr.Row():
                            with gr.Column(scale=1):
                                poet_dropdown = gr.Dropdown(
                                    choices=_poets,
                                    value=_poets[0] if _poets else None,
                                    label="Select poet",
                                    interactive=True,
                                )
                                poet_btn = gr.Button(
                                    "Show fingerprint", variant="primary"
                                )
                            with gr.Column(scale=3):
                                poet_plot = gr.Plot(
                                    label="Imagery Fingerprint",
                                    value=build_poet_fingerprint_plot(
                                        _poets[0] if _poets else ""
                                    ),
                                )
                        poet_btn.click(
                            fn=build_poet_fingerprint_plot,
                            inputs=[poet_dropdown],
                            outputs=[poet_plot],
                        )
                        poet_dropdown.change(
                            fn=build_poet_fingerprint_plot,
                            inputs=[poet_dropdown],
                            outputs=[poet_plot],
                        )

                    with gr.Tab("🗺️ Genre Heatmap"):
                        gr.HTML(
                            """<div class="small" style="margin-bottom:0.75rem;">
                            Each cell shows what percentage of clips in that genre contain each imagery tag.
                            Darker = more frequent. Hover a cell for the exact value.
                            Columns are the 25 most common tags corpus-wide; genres are rows.
                            Use this to see which imagery motifs define each genre.
                            </div>"""
                        )
                        _ = gr.Plot(
                            label="Genre × Imagery Heatmap",
                            value=build_genre_heatmap_plot(),
                        )

            gr.HTML(
                """
                <div class="card">
                  <div class="section-title">How to use this tab</div>
                  <div class="small" style="margin-bottom:0.5rem;">
                    <strong>Exploratory:</strong> Start with the Imagery Explorer above — browse poet fingerprints
                    and genre heatmaps to understand the corpus before you search.
                    No audio needed; just pick a poet or genre to see which imagery motifs define their voice.
                  </div>
                  <div class="small" style="margin-bottom:0.5rem;">
                    <strong>Search:</strong> Upload or record a clip and click <em>Search corpus</em>.
                    The system transcribes the audio, encodes it, and ranks the closest corpus poems
                    by embedding similarity. Use the genre / poet / imagery filters to narrow the
                    candidate set. The ranked results appear as a table and as numbered gold stars on
                    the corpus map below.
                  </div>
                  <div class="small">
                    <strong>Open a candidate:</strong> select a poem from the dropdown and click
                    <em>Open candidate poem</em> to see its full text, emotion breakdown, and imagery reading.
                  </div>
                </div>
                """
            )
            with gr.Row():
                search_top_k = gr.Slider(
                    1, 10, value=5, step=1, label="Number of candidate poems"
                )
                search_genre = gr.Textbox(label="Genre filter", placeholder="optional")
                search_imagery = gr.Textbox(
                    label="Imagery tag filter", placeholder="optional"
                )
                search_poet = gr.Textbox(label="Poet filter", placeholder="optional")
                search_device = gr.Radio(
                    choices=["cpu", "mps", "cuda"],
                    value=default_device,
                    label="Compute device",
                )

            with gr.Tabs():
                with gr.Tab("Upload audio"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            search_upload_audio = gr.Audio(
                                type="filepath",
                                sources=["upload"],
                                label="Upload search clip",
                            )
                            search_upload_lora = gr.Checkbox(
                                label="Use LoRA (experimental)", value=False
                            )
                            search_upload_btn = gr.Button(
                                "Search corpus with upload", variant="primary"
                            )
                        with gr.Column(scale=2):
                            search_preview = gr.HTML()
                            search_results = gr.HTML()
                            search_choice = gr.Dropdown(label="Open candidate")
                            search_open = gr.Button("Open candidate poem")
                    search_map_plot = gr.Plot(
                        label="Where similar poems sit in the corpus"
                    )
                    search_upload_btn.click(
                        fn=search_audio_candidates,
                        inputs=[
                            search_upload_audio,
                            search_top_k,
                            search_genre,
                            search_imagery,
                            search_poet,
                            search_device,
                            search_upload_lora,
                        ],
                        outputs=[search_preview, search_results, search_choice, search_map_plot],
                    )

                with gr.Tab("Record audio"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            search_record_audio = gr.Audio(
                                type="filepath",
                                sources=["microphone"],
                                label="Record search clip",
                            )
                            search_record_lora = gr.Checkbox(
                                label="Use LoRA (experimental)", value=False
                            )
                            search_record_btn = gr.Button(
                                "Search corpus with recording", variant="primary"
                            )
                        with gr.Column(scale=2):
                            search_preview_record = gr.HTML()
                            search_results_record = gr.HTML()
                            search_choice_record = gr.Dropdown(label="Open candidate")
                            search_open_record = gr.Button("Open candidate poem")
                    search_record_btn.click(
                        fn=search_audio_candidates,
                        inputs=[
                            search_record_audio,
                            search_top_k,
                            search_genre,
                            search_imagery,
                            search_poet,
                            search_device,
                            search_record_lora,
                        ],
                        outputs=[
                            search_preview_record,
                            search_results_record,
                            search_choice_record,
                            search_map_plot,
                        ],
                    )

            search_outputs = build_result_panel("Selected candidate", include_retrieval=False)
            search_open.click(
                fn=build_poem_card_from_id,
                inputs=[search_choice, search_top_k, search_imagery, search_device],
                outputs=list(search_outputs),
            )
            search_open_record.click(
                fn=build_poem_card_from_id,
                inputs=[
                    search_choice_record,
                    search_top_k,
                    search_imagery,
                    search_device,
                ],
                outputs=list(search_outputs),
            )
        # ── Button wiring ──────────────────────────────────────────────────────
        _all_screens = [start_screen, analyze_group, search_group]

        analyze_start.click(
            fn=show_analyze_mode, outputs=_all_screens, show_progress="hidden"
        )
        search_start.click(
            fn=show_search_mode, outputs=_all_screens, show_progress="hidden"
        )
        analyze_back.click(
            fn=show_start_screen, outputs=_all_screens, show_progress="hidden"
        )
        search_back.click(
            fn=show_start_screen, outputs=_all_screens, show_progress="hidden"
        )

    return ui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nabat-AI MotherDuck-style app")
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link"
    )
    parser.add_argument("--port", type=int, default=7860, help="Local port")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Default device shown in UI"
    )
    return parser.parse_args()


def choose_server_port(preferred_port: int, max_attempts: int = 10) -> int:
    for port in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred_port


if __name__ == "__main__":
    args = parse_args()
    app = build_ui(default_device=args.device)
    launch_port = choose_server_port(args.port)
    if launch_port != args.port:
        logger.warning(
            "Port {} is busy; launching on {} instead.", args.port, launch_port
        )
    logger.info("Launching app at http://127.0.0.1:{}", launch_port)
    icon_dir = PROJECT_ROOT / "src" / "icons"
    allowed = [str(icon_dir)] if icon_dir.exists() else []
    app.launch(server_port=launch_port, share=args.share, allowed_paths=allowed)
