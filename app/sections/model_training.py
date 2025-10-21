# app/sections/model_training.py
import gradio as gr
from source.train import train_model
from source.utils import config as cfg
from source.utils.carbon_utils import format_car_distance, format_car_distance_meters_only
from pathlib import Path
import pandas as pd


def load_emissions_history():
    """Load emissions history from CSV file and format it"""
    emissions_file = Path(cfg.MODEL_PATH).parent / "emissions.csv"
    if emissions_file.exists():
        try:
            df = pd.read_csv(emissions_file)
            
            # Select and rename columns with units
            column_mapping = {
                'timestamp': 'Timestamp',
                'duration': 'Duration (s)',
                'emissions': 'Emissions (kg CO₂eq)',
                'energy_consumed': 'Energy (kWh)'
            }
            
            # Get only available columns
            available_cols = [col for col in column_mapping.keys() if col in df.columns]
            df_filtered = df[available_cols].copy()
            
            # Rename columns to include units
            df_filtered.rename(columns={col: column_mapping[col] for col in available_cols}, inplace=True)
            
            # Add car distance column if emissions exist (meters only, unit in column name)
            if 'Emissions (kg CO₂eq)' in df_filtered.columns:
                df_filtered['Car Distance Equivalent (m)'] = df_filtered['Emissions (kg CO₂eq)'].apply(
                    lambda x: format_car_distance_meters_only(x) if pd.notna(x) else 'N/A'
                )
            
            # Format numeric columns
            if 'Duration (s)' in df_filtered.columns:
                df_filtered['Duration (s)'] = df_filtered['Duration (s)'].round(2)
            if 'Emissions (kg CO₂eq)' in df_filtered.columns:
                df_filtered['Emissions (kg CO₂eq)'] = df_filtered['Emissions (kg CO₂eq)'].apply(
                    lambda x: f"{x:.6f}" if pd.notna(x) else 'N/A'
                )
            if 'Energy (kWh)' in df_filtered.columns:
                df_filtered['Energy (kWh)'] = df_filtered['Energy (kWh)'].round(4)
            
            # Return last 10 trainings
            return df_filtered.tail(10)
            
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
        
        # Format metrics
        metrics_info = ""
        if result.get('metrics'):
            metrics = result['metrics']
            metrics_info = "\n\n### 📊 Training Results\n"
            if metrics.get('train_acc') is not None:
                metrics_info += f"- **Train Accuracy:** {metrics['train_acc']*100:.2f}%\n"
            if metrics.get('val_acc') is not None:
                metrics_info += f"- **Validation Accuracy:** {metrics['val_acc']*100:.2f}%\n"
        
        # Format emissions
        emissions_info = ""
        if result.get('emissions'):
            emissions_info = (
                f"\n\n### 🌍 Carbon Footprint\n"
                f"- **Emissions:** {result['emissions']['emissions_g']:.2f}g CO₂eq "
                f"({result['emissions']['emissions_kg']:.6f} kg)\n"
                f"- **🚗 Car equivalent:** {result['emissions']['car_distance_formatted']} driven\n"
            )
            if result['emissions']['duration_seconds']:
                emissions_info += f"- **Duration:** {result['emissions']['duration_seconds']:.1f}s\n"
            
            emissions_info += (
                f"\n*Based on average European car emissions of 120g CO₂/km*"
            )
        
        final_message = (
            f"✅ **Training Complete!**\n\n"
            f"### Training Configuration\n"
            f"- **Batch Size:** {batch_size}\n"
            f"- **Learning Rate:** {learning_rate}\n"
            f"- **Epochs:** {max_epochs}\n"
            f"{metrics_info}"
            f"\n### Output\n"
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
            "Carbon emissions are tracked automatically and compared to car travel distance."
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
                    label="Status",
                    height=400  # Increased height for better visibility
                )
        
        gr.Markdown("---")
        gr.Markdown("#### 📊 Training History & Carbon Footprint")
        
        # Load initial history
        initial_history = load_emissions_history()
        
        emissions_table = gr.Dataframe(
            value=initial_history,
            label="Recent Training Sessions",
            interactive=False,
            wrap=True
        )
        
        refresh_btn = gr.Button("🔄 Refresh History", size="sm")
        
        gr.Markdown("---")
        gr.Markdown(
            "**🚗 Car Distance Comparison:** Based on average European car emissions (120g CO₂/km). "
            "This helps put the carbon footprint in perspective.\n\n"
            "**Note:** Training uses the same codebase as `source/train.py`. "
            "Carbon emissions data is saved to `emissions.csv` in the model directory. "
            "Learn more about CodeCarbon at [codecarbon.io](https://codecarbon.io/)"
        )
        
        # Connect button to training function
        # Only update status output during training, emissions_table updates at the end
        train_btn.click(
            fn=run_training,
            inputs=[batch_size, learning_rate, max_epochs, track_carbon],
            outputs=[output, emissions_table]
        )
        
        # Refresh emissions history independently
        refresh_btn.click(
            fn=load_emissions_history,
            inputs=[],
            outputs=emissions_table
        )