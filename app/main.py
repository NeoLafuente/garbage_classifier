#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Gradio Application Entry Point.

This module serves as the main entry point for the Garbage Classifier
interactive demo. It orchestrates the creation of the Gradio interface,
combining data exploration, model training, and evaluation sections into
a unified web application with real-time carbon footprint tracking.

Usage
-----
Command line:
    $ uv run python app/main.py

The application will launch a Gradio interface accessible via web browser,
with an optional shareable link for remote access.

Notes
-----
The application includes:
- Data Exploration: Visualize dataset statistics and prototypes
- Model Training: Train new models with custom hyperparameters
- Model Evaluation: Evaluate models and make predictions
- Carbon Tracking: Real-time display of cumulative carbon emissions
"""
__docformat__ = "numpy"

import gradio as gr
from app.sections.data_exploration import data_exploration_tab
from app.sections.model_training import model_training_tab
from app.sections.model_evaluation import model_evaluation_tab
from source.utils import config as cfg
from source.utils.carbon_utils import format_total_emissions_display
from source.utils.config import get_valid_dir as gvd
# from pathlib import Path


def get_emissions_path():
    """
    Get the path to the emissions CSV file.

    Returns
    -------
    pathlib.Path
        Path object pointing to the emissions.csv file in the model directory.

    Notes
    -----
    The emissions file is located in the same directory as the trained model
    checkpoint, as defined in the configuration.
    """
    # return Path(cfg.MODEL_PATH).parent / "emissions.csv"
    return f"{gvd(cfg.BEST_MODEL_DIR)}/{cfg.EMISSIONS_FILE}"


def update_carbon_display():
    """
    Update the carbon footprint display with latest emissions data.

    Returns
    -------
    str
        HTML-formatted string containing the total carbon emissions statistics,
        including CO₂ equivalent mass and car distance equivalent.

    Notes
    -----
    This function reads the emissions CSV file and formats the cumulative
    carbon footprint for display in the Gradio interface header.
    """
    return format_total_emissions_display(get_emissions_path())


def main():
    """
    Launch the main Gradio application interface.

    Creates and configures a multi-tab Gradio interface with data exploration,
    model training, and evaluation capabilities. Includes a persistent carbon
    footprint counter in the header and custom CSS styling.

    The interface includes:
    - Header with project title and carbon counter
    - Tab 1: Data Exploration (dataset visualization and analysis)
    - Tab 2: Training Interface (model training with custom parameters)
    - Tab 3: Model Evaluation (metrics visualization and inference)

    Notes
    -----
    The application launches with `share=True`, which creates a public URL
    for remote access. Set to `False` for local-only access.

    Custom CSS is injected to style the carbon counter with a gradient
    background and appropriate sizing.

    See Also
    --------
    data_exploration_tab : Data exploration UI section
    model_training_tab : Model training UI section
    model_evaluation_tab : Model evaluation UI section
    """
    with gr.Blocks(title="Garbage Classifier Demo") as demo:
        # Header with carbon counter
        with gr.Row():
            gr.Markdown("# 🗑️♻️ Garbage Classifier Interactive Demo")
            carbon_display = gr.HTML(
                value=update_carbon_display(), elem_id="carbon-counter"
            )

        # Tabs
        with gr.Tabs():
            with gr.Tab("Data Exploration"):
                data_exploration_tab()

            with gr.Tab("Training Interface"):
                model_training_tab(carbon_display)

            with gr.Tab("Model Evaluation"):
                model_evaluation_tab(carbon_display)

        # Add custom CSS for the carbon counter
        demo.load(
            fn=None,
            js="""
            function() {
                const style = document.createElement('style');
                style.textContent = `
                    #carbon-counter {
                        color: white !important;
                        font-size: 1.0em;
                        font-weight: bold;
                        padding: 12px 20px;
                        background: \
                            linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 8px;
                        margin-left: auto;
                        min-width: 500px;
                        max-width: 800px;
                    }
                    #carbon-counter * {
                        color: white !important;
                    }
                `;
                document.head.appendChild(style);
            }
            """,
        )

    demo.launch(share=True)


if __name__ == "__main__":
    main()
