# app/sections/model_evaluation.py
import gradio as gr
from source.utils.carbon_utils import format_total_emissions_display
from source.utils import config as cfg
from pathlib import Path


def get_emissions_path():
    """Get the path to the emissions CSV file"""
    return Path(cfg.MODEL_PATH).parent / "emissions.csv"


def model_evaluation_tab(carbon_display):
    """
    Placeholder for Model Evaluation UI
    
    Parameters
    ----------
    carbon_display : gr.Markdown
        The carbon counter display component to update
    """
    with gr.Row():
        gr.Markdown("### ✅ Model Evaluation Section")
        gr.Markdown("This section will show model predictions and evaluation metrics.")
    
    # If you add inference here later, make sure to:
    # 1. Track carbon emissions during inference
    # 2. Return updated carbon display text
    # Example:
    # predict_btn.click(
    #     fn=run_inference,
    #     inputs=[...],
    #     outputs=[result_output, carbon_display]
    # )
    
    return []