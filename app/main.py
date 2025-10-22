# app/main.py
import gradio as gr
from app.sections.data_exploration import data_exploration_tab
from app.sections.model_training import model_training_tab
from app.sections.model_evaluation import model_evaluation_tab
from source.utils import config as cfg
from source.utils.carbon_utils import format_total_emissions_display
from pathlib import Path


def get_emissions_path():
    """Get the path to the emissions CSV file"""
    return Path(cfg.MODEL_PATH).parent / "emissions.csv"

def update_carbon_display():
    """Update the carbon footprint display"""
    return format_total_emissions_display(get_emissions_path())

def main():
    with gr.Blocks(title="Garbage Classifier Demo") as demo:
        # Header with carbon counter
        with gr.Row():
            gr.Markdown("# 🗑️♻️ Garbage Classifier Interactive Demo")
            carbon_display = gr.HTML(
                value=update_carbon_display(),
                elem_id="carbon-counter"
            )
        
        # Tabs
        with gr.Tabs():
            with gr.Tab("Data Exploration"):
                data_exploration_tab()
            
            with gr.Tab("Training Interface"):
                train_outputs = model_training_tab(carbon_display)
            
            with gr.Tab("Model Evaluation"):
                eval_outputs = model_evaluation_tab(carbon_display)
        
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
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
            """
        )

    demo.launch(share=True)


if __name__ == "__main__":
    main()