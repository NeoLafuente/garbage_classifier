# app/sections/model_training.py
import gradio as gr
from source.train import train_model
from source.utils import config as cfg
from pathlib import Path
import pandas as pd
from datetime import datetime


def load_emissions_history():
    """Load emissions history from CSV file"""
    emissions_file = Path(cfg.MODEL_PATH).parent / "emissions.csv"
    if emissions_file.exists():
        try:
            df = pd.read_csv(emissions_file)
            # Get only relevant columns
            columns_to_show = ['timestamp', 'project_name', 'duration', 'emissions', 'energy_consumed']
            available_cols = [col for col in columns_to_show if col in df.columns]
            return df[available_cols].tail(10)  # Last 10 trainings
        except Exception as e:
            return pd.DataFrame({'Error': [f'Could not load emissions data: {str(e)}']})
    return pd.DataFrame({'Info': ['No training history yet']})


def run_training(batch_size, learning_rate, max_epochs, track_carbon, progress=gr.Progress()):
    """
    Run model training with specified hyperparameters.
    
    Parameters
    ----------
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate
    max_epochs : int
        Maximum number of epochs
    track_carbon : bool
        Whether to track carbon emissions
    progress : gr.Progress
        Gradio progress tracker
        
    Returns
    -------
    tuple
        (status_message, emissions_history_dataframe)
    """
    status_messages = []
    
    def progress_callback(message):
        """Callback to update progress in Gradio UI"""
        status_messages.append(message)
        progress(len(status_messages) / (max_epochs + 3), desc=message)
    
    try:
        # Run training
        result = train_model(
            batch_size=int(batch_size),
            lr=float(learning_rate),
            max_epochs=int(max_epochs),
            track_carbon=track_carbon,
            progress_callback=progress_callback
        )
        
        emissions_info = ""
        if result['emissions']:
            emissions_info = (
                f"\n\n### 🌍 Carbon Footprint\n"
                f"- **Emissions:** {result['emissions']['emissions_g']:.2f}g CO₂eq "
                f"({result['emissions']['emissions_kg']:.6f} kg)\n"
            )
            if result['emissions']['duration_seconds']:
                emissions_info += f"- **Duration:** {result['emissions']['duration_seconds']:.1f}s\n"
        
        final_message = (
            f"✅ **Training Complete!**\n\n"
            f"### Training Configuration\n"
            f"- **Batch Size:** {batch_size}\n"
            f"- **Learning Rate:** {learning_rate}\n"
            f"- **Epochs:** {max_epochs}\n\n"
            f"### Output\n"
            f"- **Model saved at:** `{cfg.MODEL_PATH}`\n"
            f"- **Loss curves saved at:** `{cfg.LOSS_CURVES_PATH}`"
            f"{emissions_info}"
        )
        
        # Load updated emissions history
        emissions_df = load_emissions_history()
        
        return final_message, emissions_df
        
    except Exception as e:
        error_msg = f"❌ **Training Failed!**\n\n**Error:** {str(e)}"
        return error_msg, load_emissions_history()


def model_training_tab():
    """Create the Training Interface UI"""
    with gr.Column():
        gr.Markdown("### ⚙️ Model Training Interface")
        gr.Markdown(
            "Configure and train the garbage classification model with custom hyperparameters. "
            "Carbon emissions are tracked automatically."
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
                    info="Number of samples per training batch"
                )
                
                learning_rate = gr.Number(
                    value=1e-3,
                    label="Learning Rate",
                    info="Step size for optimizer (e.g., 0.001, 1e-4)"
                )
                
                max_epochs = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=cfg.MAX_EPOCHS,
                    step=1,
                    label="Max Epochs",
                    info="Number of training epochs"
                )
                
                track_carbon = gr.Checkbox(
                    value=True,
                    label="🌍 Track Carbon Emissions",
                    info="Monitor environmental impact during training"
                )
                
                train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("#### Training Status")
                output = gr.Markdown(
                    "Click 'Start Training' to begin...",
                    label="Status"
                )
        
        gr.Markdown("---")
        gr.Markdown("#### 📊 Training History & Carbon Footprint")
        
        emissions_table = gr.Dataframe(
            value=load_emissions_history(),
            label="Recent Training Sessions",
            interactive=False
        )
        
        refresh_btn = gr.Button("🔄 Refresh History", size="sm")
        
        gr.Markdown("---")
        gr.Markdown(
            "**Note:** Training uses the same codebase as `source/train.py`. "
            "Carbon emissions data is saved to `emissions.csv` in the model directory. "
            "Learn more about CodeCarbon at [codecarbon.io](https://codecarbon.io/)"
        )
        
        # Connect button to training function
        train_btn.click(
            fn=run_training,
            inputs=[batch_size, learning_rate, max_epochs, track_carbon],
            outputs=[output, emissions_table]
        )
        
        # Refresh emissions history
        refresh_btn.click(
            fn=load_emissions_history,
            inputs=[],
            outputs=emissions_table
        )