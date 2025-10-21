# app/sections/data_exploration.py

import os
import sys
import gradio as gr

# Añadir rutas al sys.path (solo una vez)
# ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../source"))
# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)

from source.utils.custom_classes.EdaAnalyzer import EdaAnalyzer


def data_exploration_tab():
    """UI for Data Exploration section."""

    # --- Inicialización única ---
    eda = EdaAnalyzer()
    eda.ensure_dataset()
    eda.load_metadata()

    # --- Variables globales del módulo ---
    cached_dir = os.path.join(os.getcwd(), "app", "sections", "cached_data")
    os.makedirs(cached_dir, exist_ok=True)
    mean_arrays_path = os.path.join(cached_dir, "mean_prototypes.npy")
    median_arrays_path = os.path.join(cached_dir, "median_prototypes.npy")

    # --- Definición de funciones auxiliares ---
    def show_random_samples():
        fig = eda.plot_random_examples_per_class()
        return fig, "✅ Random samples plotted."

    def show_mean_prototypes():
        fig = eda.plot_mean_images_per_class(filename=mean_arrays_path)
        return fig, f"✅ Mean prototypes plotted (saved in {mean_arrays_path})"

    def show_median_prototypes():
        fig = eda.plot_median_images_per_class(filename=median_arrays_path)
        return fig, f"✅ Median prototypes plotted (saved in {median_arrays_path})"

    def show_class_distribution():
        fig = eda.plot_class_distribution()
        return fig, "✅ Class distribution plotted."

    # --- Construcción de interfaz Gradio ---
    with gr.Row():
        gr.Markdown("### 📊 Data Exploration Section")
        gr.Markdown(
            "Explore dataset structure, class balance, and prototype images below."
        )

    with gr.Row():
        btn_random = gr.Button("🎲 Show Random Samples")
        btn_distribution = gr.Button("📈 Show Class Distribution")
        btn_mean = gr.Button("🧠 Show Mean Prototypes")
        btn_median = gr.Button("⚖️ Show Median Prototypes")

    output_plot = gr.Plot(label="Visualization")
    output_text = gr.Textbox(label="Status", interactive=False)

    # --- Lógica de los botones ---
    btn_random.click(fn=show_random_samples, outputs=[output_plot, output_text])
    btn_distribution.click(fn=show_class_distribution, outputs=[output_plot, output_text])
    btn_mean.click(fn=show_mean_prototypes, outputs=[output_plot, output_text])
    btn_median.click(fn=show_median_prototypes, outputs=[output_plot, output_text])

    return [btn_random, btn_distribution, btn_mean, btn_median, output_plot, output_text]
