#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Training Interface for Gradio Application.

This module provides the Gradio UI components for training the garbage
classification model with custom hyperparameters. It includes real-time
progress tracking, carbon emissions monitoring, and training history
visualization.

The interface allows users to:
- Configure training hyperparameters (batch size, learning rate, epochs)
- Track carbon emissions during training
- View training results and metrics
- Review training history with environmental impact metrics

Notes
-----
Training emissions are automatically tracked using CodeCarbon and stored
in a CSV file alongside the model checkpoint. The interface displays
emissions as both absolute values (kg CO₂eq) and relative comparisons
(equivalent car driving distance).
"""
__docformat__ = "numpy"

import gradio as gr
from source.train import train_model
from source.utils import config as cfg
from source.utils.config import get_valid_dir as gvd
from source.utils.carbon_utils import (
    format_car_distance_meters_only,
    format_total_emissions_display,
)
import pandas as pd


def load_emissions_history():
    """
    Load and format emissions history from CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the last 10 training sessions with formatted
        columns including timestamp, duration, emissions, energy consumption,
        and car distance equivalent. Returns error/info message if file
        cannot be loaded or doesn't exist.

    Notes
    -----
    The function performs the following transformations:
    - Selects relevant columns (timestamp, duration, emissions, energy)
    - Adds units to column names for clarity
    - Calculates car distance equivalent in meters
    - Formats numeric values with appropriate precision
    - Returns only the most recent 10 training sessions

    If the emissions file doesn't exist or cannot be read, returns a
    DataFrame with an appropriate message.
    """
    emissions_file = cfg.get_emissions_path()
    if emissions_file.exists():
        try:
            df = pd.read_csv(emissions_file)

            # Only show trainig emissions
            if "project_name" in df.columns:
                df = df[df["project_name"] == "garbage_classifier_training"].copy()

            if df.empty:
                return pd.DataFrame({"Info": ["No training history yet"]})

            # Select and rename columns with units
            column_mapping = {
                "timestamp": "Timestamp",
                "duration": "Duration (s)",
                "emissions": "Emissions (kg CO₂eq)",
                "energy_consumed": "Energy (kWh)",
            }

            # Get only available columns
            available_cols = [
                col for col in column_mapping.keys() if col in df.columns
            ]
            df_filtered = df[available_cols].copy()

            # Rename columns to include units
            df_filtered.rename(
                columns={col: column_mapping[col] for col in available_cols},
                inplace=True,
            )

            # Add car distance column if emissions exist
            if "Emissions (kg CO₂eq)" in df_filtered.columns:
                df_filtered["Car Distance Equivalent (m)"] = df_filtered[
                    "Emissions (kg CO₂eq)"
                ].apply(
                    lambda x: (
                        format_car_distance_meters_only(x)
                        if pd.notna(x) else "N/A"
                    )
                )

            # Format numeric columns
            if "Duration (s)" in df_filtered.columns:
                df_filtered["Duration (s)"] = \
                    df_filtered["Duration (s)"].round(2)
            if "Emissions (kg CO₂eq)" in df_filtered.columns:
                df_filtered["Emissions (kg CO₂eq)"] = df_filtered[
                    "Emissions (kg CO₂eq)"
                ].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
            if "Energy (kWh)" in df_filtered.columns:
                df_filtered["Energy (kWh)"] = \
                    df_filtered["Energy (kWh)"].round(4)

            # Return last 10 trainings
            return df_filtered.tail(10)

        except Exception as e:
            return pd.DataFrame(
                {"Error": [f"Could not load emissions data: {str(e)}"]}
            )
    return pd.DataFrame({"Info": ["No training history yet"]})


def run_training(
    batch_size, learning_rate, max_epochs, track_carbon, progress=gr.Progress()
):
    """
    Execute model training with specified hyperparameters.

    This function serves as the Gradio wrapper for the training process,
    handling progress updates, result formatting, and UI state management.

    Parameters
    ----------
    batch_size : int
        Number of samples per training batch. Larger values use more memory
        but may improve training stability.
    learning_rate : float
        Step size for the optimizer (e.g., 0.001, 1e-4). Controls how much
        model weights are updated during training.
    max_epochs : int
        Maximum number of complete passes through the training dataset.
    track_carbon : bool
        Whether to track and record carbon emissions during training using
        CodeCarbon.
    progress : gr.Progress, optional
        Gradio progress tracker for updating the UI progress bar and status
        messages.

    Returns
    -------
    tuple of (str, pd.DataFrame, str)
        - status_message : Markdown-formatted string with training results,
          including metrics and carbon footprint information
        - emissions_history : Updated DataFrame with training history
        - carbon_display : Updated HTML string for the carbon counter display

    Notes
    -----
    The function includes a progress callback that updates the Gradio UI
    in real-time during training. Training results include:
    - Training and validation accuracy
    - Model checkpoint location
    - Loss curves location
    - Carbon emissions (if tracked)
    - Car distance equivalent

    If training fails, returns an error message while preserving the
    existing emissions history and carbon display.

    See Also
    --------
    source.train.train_model : Core training function
    load_emissions_history : Load training history from CSV
    """
    status_messages = []

    def progress_callback(message):
        """
        Callback to update progress in Gradio UI.

        Parameters
        ----------
        message : str
            Progress message to display
        """
        status_messages.append(message)
        progress(len(status_messages) / (max_epochs + 3), desc=message)

    try:
        # Run training
        result = train_model(
            batch_size=int(batch_size),
            lr=float(learning_rate),
            max_epochs=int(max_epochs),
            track_carbon=track_carbon,
            progress_callback=progress_callback,
        )

        # Format metrics
        metrics_info = ""
        if result.get("metrics"):
            metrics = result["metrics"]
            metrics_info = "\n\n### 📊 Training Results\n"
            if metrics.get("train_acc") is not None:
                metrics_info += (
                    f"- **Train Accuracy:** {metrics['train_acc']*100:.2f}%\n"
                )
            if metrics.get("val_acc") is not None:
                metrics_info += (
                    f"- **Validation Accuracy:** \
                        {metrics['val_acc']*100:.2f}%\n"
                )

        # Format emissions
        emissions_info = ""
        if result.get("emissions"):
            emissions_info = (
                f"\n\n### 🌍 Carbon Footprint\n"
                f"- **Emissions:** \
                    {result['emissions']['emissions_g']:.2f}g CO₂eq "
                f"({result['emissions']['emissions_kg']:.6f} kg)\n"
                f"- **🚗 Car equivalent:** \
                    {result['emissions']['car_distance_formatted']} driven\n"
            )
            if result["emissions"]["duration_seconds"]:
                emissions_info += (
                    f"- **Duration:** \
                        {result['emissions']['duration_seconds']:.1f}s\n"
                )

            emissions_info += (
                "\n*Based on average European car emissions of 120g CO₂/km*"
            )

        model_save_path = f"{gvd(cfg.WEIGHTS_DIR)}/{cfg.MODEL_FILENAME}" #"model_resnet18_garbage.ckpt"
        loss_curves_path = gvd(cfg.LOSS_CURVES_PATH)

        final_message = (
            f"✅ **Training Complete!**\n\n"
            f"### Training Configuration\n"
            f"- **Batch Size:** {batch_size}\n"
            f"- **Learning Rate:** {learning_rate}\n"
            f"- **Epochs:** {max_epochs}\n"
            f"{metrics_info}"
            f"\n### Output\n"
            f"- **Model saved at:** `{model_save_path}`\n"
            f"- **Loss curves saved at:** `{loss_curves_path}`"
            f"{emissions_info}"
        )

        # Load updated emissions history
        emissions_df = load_emissions_history()

        # Update carbon display
        carbon_text = format_total_emissions_display(cfg.get_emissions_path())

        return final_message, emissions_df, carbon_text

    except Exception as e:
        error_msg = f"❌ **Training Failed!**\n\n**Error:** {str(e)}"
        carbon_text = format_total_emissions_display(cfg.get_emissions_path())
        return error_msg, load_emissions_history(), carbon_text


def model_training_tab(carbon_display):
    """
    Create the Training Interface UI section.

    Builds a Gradio interface for configuring and launching model training,
    with real-time progress tracking, emissions monitoring, and training
    history visualization.

    Parameters
    ----------
    carbon_display : gr.HTML
        The carbon counter display component to update after training
        completes. This component shows cumulative emissions across all
        training sessions.

    Returns
    -------
    list of gr.Component
        List containing [output, emissions_table] components for potential
        external reference.

    Notes
    -----
    The interface is organized into sections:

    1. **Hyperparameters Panel:**
       - Batch size slider (8-128)
       - Learning rate input
       - Max epochs slider (1-100)
       - Carbon tracking checkbox
       - Start training button

    2. **Status Panel:**
       - Real-time training progress
       - Training results and metrics
       - Carbon footprint statistics

    3. **History Panel:**
       - Recent training sessions table
       - Emissions and energy consumption
       - Car distance equivalents
       - Refresh button

    The carbon tracking uses CodeCarbon and compares emissions to average
    European car travel (120g CO₂/km).

    Examples
    --------
    >>> with gr.Blocks() as demo:
    ...     carbon_display = gr.HTML()
    ...     model_training_tab(carbon_display)
    """
    with gr.Column():
        gr.Markdown("### ⚙️ Model Training Interface")
        gr.Markdown(
            "Configure and train the garbage \
                classification model with custom hyperparameters. "
            "Carbon emissions are tracked \
                automatically and compared to car travel distance."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Training Hyperparameters")

                batch_size = gr.Slider(
                    minimum=8,
                    maximum=128,
                    value=32,
                    step=8,
                    label="Batch Size",
                    info="Number of samples per training batch",
                )

                learning_rate = gr.Number(
                    value=1e-3,
                    label="Learning Rate",
                    info="Step size for optimizer (e.g., 0.001, 1e-4)",
                )

                max_epochs = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=cfg.MAX_EPOCHS,
                    step=1,
                    label="Max Epochs",
                    info="Number of training epochs",
                )

                track_carbon = gr.Checkbox(
                    value=True,
                    label="🌍 Track Carbon Emissions",
                    info="Monitor environmental impact during training",
                )

                train_btn = gr.Button(
                    "🚀 Start Training", variant="primary", size="lg"
                )

            with gr.Column(scale=1):
                gr.Markdown("#### Training Status")
                output = gr.Markdown(
                    "Click 'Start Training' to begin...",
                    label="Status", height=400
                )

        gr.Markdown("---")
        gr.Markdown("#### 📊 Training History & Carbon Footprint")

        # Load initial history
        initial_history = load_emissions_history()

        emissions_table = gr.Dataframe(
            value=initial_history,
            label="Recent Training Sessions",
            interactive=False,
            wrap=True,
        )

        refresh_btn = gr.Button("🔄 Refresh History", size="sm")

        gr.Markdown("---")
        gr.Markdown(
            "**🚗 Car Distance Comparison:** Based on average \
                European car emissions (120g CO₂/km). "
            "Carbon Footprint was calculated using CodeCarbon. "
            "Learn more about CodeCarbon at [codecarbon.io]\
                (https://codecarbon.io/)"
        )

        # Connect button to training function - now updates carbon display too
        train_btn.click(
            fn=run_training,
            inputs=[batch_size, learning_rate, max_epochs, track_carbon],
            outputs=[output, emissions_table, carbon_display],
        )

        # Refresh emissions history independently
        refresh_btn.click(
            fn=load_emissions_history, inputs=[], outputs=emissions_table
        )

    return [output, emissions_table]
