import os
import sys
import gradio as gr

from source.utils.custom_classes.EdaAnalyzer import EdaAnalyzer


def data_exploration_tab():
    """UI for Data Exploration section."""

    eda = EdaAnalyzer()
    eda.ensure_dataset()
    eda.load_metadata()

    cached_dir = os.path.join(os.getcwd(), "app", "sections", "cached_data")
    os.makedirs(cached_dir, exist_ok=True)
    mean_arrays_path = os.path.join(cached_dir, "mean_prototypes.npy")

    # --- Internal functions ---
    def show_random_samples():
        fig = eda.plot_random_examples_per_class()
        return fig, "✅ Random samples plotted.", gr.update(visible=False), gr.update(visible=False)

    def show_class_distribution():
        fig = eda.plot_class_distribution()
        return fig, "✅ Class distribution plotted.", gr.update(visible=False), gr.update(visible=False)

    def show_mean_prototypes():
        fig = eda.plot_mean_images_per_class(filename=mean_arrays_path)
        msg = "✅ Mean prototypes plotted. Enable Otsu binarization if you want to adjust."
        return fig, msg, gr.update(visible=True), gr.update(visible=False)

    def toggle_mean_otsu_binarization(use_otsu, threshold):
        """Toggle between normal visualization and Otsu binarization."""
        if use_otsu:
            fig = eda.plot_mean_images_per_class_with_otsu(
                threshold=threshold, filename=mean_arrays_path
            )
        else:
            fig = eda.plot_mean_images_per_class(filename=mean_arrays_path)
        return fig, gr.update(visible=use_otsu, interactive=use_otsu)

    def update_mean_otsu_threshold(threshold):
        """Update plot when slider moves (only when Otsu is enabled)."""
        fig = eda.plot_mean_images_per_class_with_otsu(
            threshold=threshold, filename=mean_arrays_path
        )
        return fig
    
    # --- UI Layout ---
    with gr.Row():
        gr.Markdown("### 📊 Data Exploration Section")
        gr.Markdown(
            "Explore dataset structure, class balance, and prototype images below."
        )

    with gr.Row():
        btn_random = gr.Button("🎲 Show Random Samples")
        btn_distribution = gr.Button("📈 Show Class Distribution")
        btn_mean = gr.Button("⚖️ Show Mean Prototypes")

    output_plot = gr.Plot(label="Visualization")
    output_text = gr.Textbox(label="Status", interactive=False)

    # --- Checkbox to enable/disable Otsu (initially hidden) ---
    with gr.Row(visible=False) as otsu_mean_controls_row:
        otsu_mean_checkbox = gr.Checkbox(
            label="🔲 Apply Otsu binarization to means",
            value=False,
            interactive=True,
        )
        mean_threshold_slider = gr.Slider(
            minimum=-1.0,
            maximum=1.0,
            value=0.0,
            step=0.05,
            label="🔧 Adjust Otsu Threshold",
            visible=False,
            interactive=False,
        )

    # --- Button interactions ---
    btn_random.click(
        fn=show_random_samples,
        outputs=[output_plot, output_text, otsu_mean_controls_row]
    )
    btn_distribution.click(
        fn=show_class_distribution,
        outputs=[output_plot, output_text, otsu_mean_controls_row]
    )

    # ============================= MEAN ==================================
    btn_mean.click(
        fn=show_mean_prototypes,
        outputs=[output_plot, output_text, otsu_mean_controls_row]
    )

    otsu_mean_checkbox.change(
        fn=toggle_mean_otsu_binarization,
        inputs=[otsu_mean_checkbox, mean_threshold_slider],
        outputs=[output_plot, mean_threshold_slider],
    )

    mean_threshold_slider.change(
        fn=update_mean_otsu_threshold,
        inputs=mean_threshold_slider,
        outputs=output_plot,
    )

    return [
        btn_random,
        btn_distribution,
        btn_mean,
        otsu_mean_controls_row,
        otsu_mean_checkbox,
        mean_threshold_slider,
        output_plot,
        output_text,
    ]