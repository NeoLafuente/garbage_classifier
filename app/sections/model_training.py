# app/sections/model_training.py
import gradio as gr
from source.train import train_model
from source.utils import config as cfg
import threading


def run_training(batch_size, learning_rate, max_epochs, progress=gr.Progress()):
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
    progress : gr.Progress
        Gradio progress tracker
        
    Returns
    -------
    str
        Training completion message
    """
    status_messages = []
    
    def progress_callback(message):
        """Callback to update progress in Gradio UI"""
        status_messages.append(message)
        progress(len(status_messages) / (max_epochs + 3), desc=message)
    
    try:
        # Run training
        trainer, model, data_module = train_model(
            batch_size=int(batch_size),
            lr=float(learning_rate),
            max_epochs=int(max_epochs),
            progress_callback=progress_callback
        )
        
        final_message = (
            f"✅ **Training Complete!**\n\n"
            f"- Batch Size: {batch_size}\n"
            f"- Learning Rate: {learning_rate}\n"
            f"- Epochs: {max_epochs}\n"
            f"- Model saved at: {cfg.MODEL_PATH}\n"
            f"- Loss curves saved at: {cfg.LOSS_CURVES_PATH}"
        )
        return final_message
        
    except Exception as e:
        return f"❌ **Training Failed!**\n\nError: {str(e)}"


def model_training_tab():
    """Create the Training Interface UI"""
    with gr.Column():
        gr.Markdown("### ⚙️ Model Training Interface")
        gr.Markdown(
            "Configure and train the garbage classification model with custom hyperparameters."
        )
        
        with gr.Row():
            with gr.Column():
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
                
                train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("#### Training Status")
                output = gr.Markdown(
                    "Click 'Start Training' to begin...",
                    label="Status"
                )
        
        # Connect button to training function
        train_btn.click(
            fn=run_training,
            inputs=[batch_size, learning_rate, max_epochs],
            outputs=output
        )