# app/main.py
import gradio as gr
from app.sections.data_exploration import data_exploration_tab
from app.sections.model_training import model_training_tab
from app.sections.model_evaluation import model_evaluation_tab

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# 🗑️♻️ Garbage Classifier Interactive Demo")

        with gr.Tab("Data Exploration"):
            data_exploration_tab()

        with gr.Tab("Training Interface"):
            model_training_tab()

        with gr.Tab("Model Evaluation"):
            model_evaluation_tab()

    demo.launch(share = True)

if __name__ == "__main__":
    main()