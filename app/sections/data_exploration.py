#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Exploration Interface for Gradio Application.

This module provides interactive visualization tools for exploring the
garbage classification dataset. It enables users to view random samples,
analyze class distributions, and examine prototype images (mean/median)
with optional Otsu binarization for enhanced visualization.

The interface includes:
- Random sample visualization per class
- Class distribution analysis
- Mean prototype images with Otsu thresholding
- Interactive controls for threshold adjustment

Notes
-----
Prototype arrays (mean/median) are cached to disk in
`app/sections/cached_data/` to avoid recomputation.
The Otsu binarization feature helps visualize the dominant
features of each class by thresholding the prototype images.
"""
__docformat__ = "numpy"

import os
import gradio as gr

from source.utils.custom_classes.EdaAnalyzer import EdaAnalyzer


def data_exploration_tab():
    """
    Create the Data Exploration UI section.

    Builds an interactive Gradio interface for exploring the dataset through
    various visualization methods. Includes buttons for different
    visualization types and dynamic controls for prototype image processing.

    Returns
    -------
    list of gr.Component
        List of all Gradio components created in this tab, including buttons,
        plots, checkboxes, and sliders. Returned for potential external
        reference or testing purposes.

    Notes
    -----
    The interface workflow:

    1. **Random Samples**: Display random images from each class
    2. **Class Distribution**: Show bar chart of samples per class
    3. **Mean Prototypes**: Display average image per class with optional
       Otsu binarization

    Otsu Controls:
    - Only visible after generating mean prototypes
    - Checkbox enables/disables binarization
    - Slider adjusts threshold (-1.0 to 1.0) when enabled
    - Updates visualization in real-time

    The EdaAnalyzer handles dataset loading, metadata management, and
    actual visualization generation. This function provides the UI wrapper.

    Cache Directory:
    All prototype arrays are saved to `app/sections/cached_data/` to enable
    fast toggling between normal and Otsu-binarized views.

    Examples
    --------
    >>> with gr.Blocks() as demo:
    ...     components = data_exploration_tab()
    ...     # components[0] is btn_random
    ...     # components[-2] is output_plot
    ...     # components[-1] is output_text

    See Also
    --------
    EdaAnalyzer : Core class for dataset analysis and visualization
    """

    eda = EdaAnalyzer()
    eda.ensure_dataset()
    eda.load_metadata()

    cached_dir = os.path.join(os.getcwd(), "app", "sections", "cached_data")
    os.makedirs(cached_dir, exist_ok=True)
    mean_arrays_path = os.path.join(cached_dir, "mean_prototypes.npy")

    # --- Internal functions ---
    def show_random_samples():
        """
        Display random sample images from each class.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, str, gr.update, gr.update)
            - Figure: Plot showing random samples per class
            - str: Status message
            - gr.update: Hide mean Otsu controls
            - gr.update: Hide median Otsu controls (commented out)

        Notes
        -----
        Generates a grid of random images, typically showing 3-5 examples
        per class to give an overview of dataset variety.
        """
        fig = eda.plot_random_examples_per_class()
        return (
            fig,
            "✅ Random samples plotted.",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def show_class_distribution():
        """
        Display bar chart of samples per class.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, str, gr.update, gr.update)
            - Figure: Bar chart showing class distribution
            - str: Status message
            - gr.update: Hide mean Otsu controls
            - gr.update: Hide median Otsu controls (commented out)

        Notes
        -----
        Helps identify class imbalance in the dataset. Each bar represents
        the number of images in a garbage category.
        """
        fig = eda.plot_class_distribution()
        return (
            fig,
            "✅ Class distribution plotted.",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def show_mean_prototypes():
        """
        Display mean prototype images for each class.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, str, gr.update, gr.update)
            - Figure: Grid of mean prototype images
            - str: Status message indicating Otsu controls are available
            - gr.update: Show mean Otsu controls
            - gr.update: Hide median Otsu controls (commented out)

        Notes
        -----
        Mean prototypes are computed by averaging all images in each class.
        They reveal the "typical" appearance and dominant features of each
        category. Cached to disk for fast subsequent access.

        After display, Otsu binarization controls become visible for
        enhanced visualization of class features.
        """
        fig = eda.plot_mean_images_per_class(filename=mean_arrays_path)
        msg = (
            "✅ Mean prototypes plotted. Enable Otsu binarization if "
            "you want to adjust."
        )
        return fig, msg, gr.update(visible=True), gr.update(visible=False)

    def toggle_mean_otsu_binarization(use_otsu, threshold):
        """
        Toggle between normal and Otsu-binarized mean prototypes.

        Parameters
        ----------
        use_otsu : bool
            If True, apply Otsu binarization; if False, show normal prototypes.
        threshold : float
            Adjustment to Otsu threshold in range [-1.0, 1.0].

        Returns
        -------
        tuple of (matplotlib.figure.Figure, gr.update)
            - Figure: Updated plot (normal or binarized)
            - gr.update: Show/hide threshold slider based on use_otsu

        Notes
        -----
        Otsu binarization converts grayscale images to binary (black/white)
        using an automatic threshold. The threshold parameter allows fine-
        tuning this threshold to emphasize different features.

        The slider is only visible and interactive when Otsu is enabled.
        """
        if use_otsu:
            fig = eda.plot_mean_images_per_class_with_otsu(
                threshold=threshold, filename=mean_arrays_path
            )
        else:
            fig = eda.plot_mean_images_per_class(filename=mean_arrays_path)
        return fig, gr.update(visible=use_otsu, interactive=use_otsu)

    def update_mean_otsu_threshold(threshold):
        """
        Update Otsu-binarized plot when threshold slider changes.

        Parameters
        ----------
        threshold : float
            New threshold adjustment value in range [-1.0, 1.0].

        Returns
        -------
        matplotlib.figure.Figure
            Updated plot with new Otsu threshold applied.

        Notes
        -----
        This function is only called when Otsu binarization is enabled.
        It allows real-time adjustment of the threshold to find the optimal
        visualization for identifying class features.

        Positive thresholds make the image darker (more black pixels),
        while negative thresholds make it lighter (more white pixels).
        """
        fig = eda.plot_mean_images_per_class_with_otsu(
            threshold=threshold, filename=mean_arrays_path
        )
        return fig

    # --- UI Layout ---
    with gr.Row():
        gr.Markdown("### 📊 Data Exploration Section")
        gr.Markdown(
            "Explore dataset structure, class balance, \
                and prototype images below."
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
        outputs=[
            output_plot,
            output_text,
            otsu_mean_controls_row,
        ],
    )
    btn_distribution.click(
        fn=show_class_distribution,
        outputs=[
            output_plot,
            output_text,
            otsu_mean_controls_row,
        ],
    )

    btn_mean.click(
        fn=show_mean_prototypes,
        outputs=[
            output_plot,
            output_text,
            otsu_mean_controls_row,
        ],
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
