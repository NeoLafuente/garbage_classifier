# app/sections/model_evaluation.py
import gradio as gr
from source.utils.carbon_utils import format_total_emissions_display
from source.utils import config as cfg
from source.utils.custom_classes.EvalAnalyzer import GarbageModelAnalyzer
from source.predict import predict_image, predict_batch, load_model_for_inference
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd


def get_emissions_path():
    """Get the path to the emissions CSV file"""
    return Path(cfg.MODEL_PATH).parent / "emissions.csv"


def get_available_models():
    """Get list of available trained models"""
    models_dict = {
        "Best Model (Provided)": str(Path("models/best/model_resnet18_garbage.ckpt")),
        "Latest Trained Model": str(cfg.MODEL_PATH)
    }
    return models_dict


# ========================
# METRICS FUNCTIONS (keep these - they're UI-specific)
# ========================

def generate_confusion_matrix(model_choice, show_normalized, progress=gr.Progress()):
    """Generate confusion matrix plot"""
    if not model_choice:
        return None, "Please select a model first"
    
    try:
        progress(0.1, desc="Loading model...")
        analyzer = GarbageModelAnalyzer()
        
        models_dict = get_available_models()
        model_path = models_dict.get(model_choice)
        
        analyzer.load_model(checkpoint_path=model_path)
        
        progress(0.3, desc="Setting up data...")
        analyzer.setup_data(batch_size=32)
        
        progress(0.5, desc="Evaluating model...")
        val_loader = analyzer.data_module.val_dataloader()
        preds, labels, probs = analyzer.evaluate_loader(val_loader)
        
        progress(0.8, desc="Generating confusion matrix...")
        
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        num_classes = cfg.NUM_CLASSES
        cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=range(num_classes))
        
        if show_normalized:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "Normalized Confusion Matrix - Validation Set"
        else:
            title = "Confusion Matrix - Validation Set"
        
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg.CLASS_NAMES)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        
        progress(1.0, desc="Done!")
        return fig, "✅ Confusion matrix generated successfully"
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# app/sections/model_evaluation.py

def get_metrics_path_for_model(model_choice):
    """Get the correct metrics.json path for the selected model"""
    if model_choice == "Best Model (Provided)":
        # Buscar en la carpeta del mejor modelo
        return Path("models/best/performance/loss_curves/metrics.json")
    else:
        # Buscar en la carpeta del último modelo entrenado
        return Path(cfg.LOSS_CURVES_PATH) / "metrics.json"


def load_loss_curves(model_choice):
    """Load and plot loss curves from metrics.json"""
    try:
        metrics_path = get_metrics_path_for_model(model_choice)
        
        if not metrics_path.exists():
            return None, f"❌ No training metrics found for {model_choice}. Path: {metrics_path}"
        
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        
        train_losses = data.get('train_losses', [])
        val_losses = data.get('val_losses', [])
        
        if not train_losses and not val_losses:
            return None, "❌ No loss data available"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        if train_losses:
            ax.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
        if val_losses:
            ax.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Loss Curves - {model_choice}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig, f"✅ Loss curves loaded successfully from {model_choice}"
        
    except Exception as e:
        return None, f"❌ Error loading loss curves: {str(e)}"


def load_accuracy_curves(model_choice):
    """Load and plot accuracy curves from metrics.json"""
    try:
        metrics_path = get_metrics_path_for_model(model_choice)
        
        if not metrics_path.exists():
            return None, f"❌ No training metrics found for {model_choice}. Path: {metrics_path}"
        
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        
        train_accs = data.get('train_accs', [])
        val_accs = data.get('val_accs', [])
        
        if not train_accs and not val_accs:
            return None, "❌ No accuracy data available"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if train_accs:
            ax.plot(range(1, len(train_accs) + 1), train_accs, 'b-o', label='Train Accuracy', linewidth=2)
        if val_accs:
            ax.plot(range(1, len(val_accs) + 1), val_accs, 'r-s', label='Validation Accuracy', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Accuracy Curves - {model_choice}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        plt.tight_layout()
        
        return fig, f"✅ Accuracy curves loaded successfully from {model_choice}"
        
    except Exception as e:
        return None, f"❌ Error loading accuracy curves: {str(e)}"


def generate_calibration_curves(model_choice, progress=gr.Progress()):
    """Generate calibration curves"""
    if not model_choice:
        return None, "Please select a model first"
    
    try:
        progress(0.1, desc="Loading model...")
        analyzer = GarbageModelAnalyzer()
        
        models_dict = get_available_models()
        model_path = models_dict.get(model_choice)
        
        analyzer.load_model(checkpoint_path=model_path)
        
        progress(0.3, desc="Setting up data...")
        analyzer.setup_data(batch_size=32)
        
        progress(0.5, desc="Evaluating model...")
        val_loader = analyzer.data_module.val_dataloader()
        preds, labels, probs = analyzer.evaluate_loader(val_loader)
        
        progress(0.8, desc="Generating calibration curves...")
        
        analyzer.plot_calibration_curves(labels, probs)
        fig = plt.gcf()
        
        progress(1.0, desc="Done!")
        return fig, "✅ Calibration curves generated successfully"
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ========================
# PREDICTION FUNCTIONS (use source/predict.py)
# ========================

def predict_single_image_gradio(model_choice, image, carbon_display_text, track_carbon=True):
    """Gradio wrapper for single image prediction"""
    if image is None:
        return None, "Please upload an image", carbon_display_text
    
    if not model_choice:
        return None, "Please select a model first", carbon_display_text
    
    try:
        # Get model path
        models_dict = get_available_models()
        model_path = models_dict.get(model_choice)
        
        # Load model once
        model, device, transform = load_model_for_inference(model_path=model_path)
        
        # Predict
        result = predict_image(
            image,
            model=model,
            transform=transform,
            device=device,
            track_carbon=track_carbon
        )
        
        # Create probabilities bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        probs_list = [result['probabilities'][cls] for cls in cfg.CLASS_NAMES]
        pred_idx = result['predicted_idx']
        colors = ['green' if i == pred_idx else 'skyblue' for i in range(len(cfg.CLASS_NAMES))]
        bars = ax.barh(cfg.CLASS_NAMES, probs_list, color=colors)
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title(f'Prediction Probabilities\nPredicted Class: {result["predicted_class"]}', 
                     fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        
        for i, (bar, prob) in enumerate(zip(bars, probs_list)):
            ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{prob*100:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Format result text
        result_text = f"### 🎯 Prediction: **{result['predicted_class']}**\n\n"
        result_text += f"**Confidence:** {result['confidence']*100:.2f}%\n\n"
        result_text += "**All Probabilities:**\n"
        for class_name, prob in result['probabilities'].items():
            emoji = "🏆" if class_name == result['predicted_class'] else "  "
            result_text += f"{emoji} {class_name}: {prob*100:.2f}%\n"
        
        # Add emissions if tracked
        updated_carbon_display = carbon_display_text
        if result['emissions']:
            emissions = result['emissions']
            result_text += f"\n\n### 🌍 Carbon Footprint\n"
            result_text += f"- **Emissions:** {emissions['emissions_g']:.4f}g CO₂eq\n"
            result_text += f"- **🚗 Car equivalent:** {emissions['car_distance_formatted']} driven\n"
            updated_carbon_display = format_total_emissions_display(get_emissions_path())
        
        return fig, result_text, updated_carbon_display
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}", carbon_display_text


def predict_folder_gradio(model_choice, files, carbon_display_text, track_carbon=True):
    """Gradio wrapper for batch prediction"""
    if not files or len(files) == 0:
        return None, "Please upload images", carbon_display_text
    
    if not model_choice:
        return None, "Please select a model first", carbon_display_text
    
    try:
        # Get model path
        models_dict = get_available_models()
        model_path = models_dict.get(model_choice)
        
        # Load model once
        model, device, transform = load_model_for_inference(model_path=model_path)
        
        # Extract file paths
        image_paths = [file.name for file in files]
        
        # USE IMPORTED FUNCTION
        batch_result = predict_batch(
            image_paths,
            model=model,
            transform=transform,
            device=device,
            track_carbon=track_carbon
        )
        
        # Format results as DataFrame
        df_data = []
        for result in batch_result['results']:
            if result['status'] == 'success':
                df_data.append({
                    'Filename': result['filename'],
                    'Predicted Class': result['predicted_class'],
                    'Confidence (%)': f"{result['confidence']*100:.2f}"
                })
            else:
                df_data.append({
                    'Filename': result['filename'],
                    'Predicted Class': 'Error',
                    'Confidence (%)': result['error']
                })
        
        df_results = pd.DataFrame(df_data)
        
        # Format result text
        summary = batch_result['summary']
        result_text = f"### 📊 Batch Prediction Results\n\n"
        result_text += f"**Total images processed:** {summary['total_images']}\n"
        result_text += f"**Successful predictions:** {summary['successful']}\n\n"
        
        # Add emissions if tracked
        updated_carbon_display = carbon_display_text
        if batch_result['emissions']:
            emissions = batch_result['emissions']
            result_text += f"### 🌍 Carbon Footprint\n"
            result_text += f"- **Emissions:** {emissions['emissions_g']:.4f}g CO₂eq\n"
            result_text += f"- **🚗 Car equivalent:** {emissions['car_distance_formatted']} driven\n"
            result_text += f"- **Avg per image:** {emissions['emissions_per_image_g']:.4f}g CO₂eq\n"
            updated_carbon_display = format_total_emissions_display(get_emissions_path())
        
        return df_results, result_text, updated_carbon_display
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}", carbon_display_text  # ✅


# ========================
# UI LAYOUT (keep as is)
# ========================

def model_evaluation_tab(carbon_display):
    """Create the Model Evaluation UI"""
    # [Keep your existing UI code, just replace the function calls]
    with gr.Column():
        gr.Markdown("### 🔬 Model Evaluation & Inference")
        gr.Markdown(
            "Evaluate trained models, visualize metrics, and make predictions on new images."
        )
        
        # Model Selection
        gr.Markdown("#### 🧠 Model Selection")
        model_choice = gr.Radio(
            choices=list(get_available_models().keys()),
            value=list(get_available_models().keys())[0],
            label="Select Model",
            info="Choose between the best provided model or your latest trained model"
        )
        
        gr.Markdown("---")
        
        # Metrics Section
        gr.Markdown("#### 📈 Model Metrics & Visualizations")
        
        with gr.Tabs():
            with gr.Tab("Confusion Matrix"):
                show_normalized = gr.Checkbox(
                    label="Show Normalized",
                    value=False,
                    info="Toggle between raw counts and normalized percentages"
                )
                cm_button = gr.Button("Generate Confusion Matrix", variant="primary")
                cm_plot = gr.Plot(label="Confusion Matrix")
                cm_status = gr.Markdown("")
                
                cm_button.click(
                    fn=generate_confusion_matrix,
                    inputs=[model_choice, show_normalized],
                    outputs=[cm_plot, cm_status]
                )
            
            with gr.Tab("Loss Curves"):
                loss_button = gr.Button("Load Loss Curves", variant="primary")
                loss_plot = gr.Plot(label="Loss Curves")
                loss_status = gr.Markdown("")
                
                loss_button.click(
                    fn=load_loss_curves,
                    inputs=[model_choice],
                    outputs=[loss_plot, loss_status]
                )
            
            with gr.Tab("Accuracy Curves"):
                acc_button = gr.Button("Load Accuracy Curves", variant="primary")
                acc_plot = gr.Plot(label="Accuracy Curves")
                acc_status = gr.Markdown("")
                
                acc_button.click(
                    fn=load_accuracy_curves,
                    inputs=[model_choice],
                    outputs=[acc_plot, acc_status]
                )
            
            with gr.Tab("Calibration Curves"):
                calib_button = gr.Button("Generate Calibration Curves", variant="primary")
                calib_plot = gr.Plot(label="Calibration Curves")
                calib_status = gr.Markdown("")
                
                calib_button.click(
                    fn=generate_calibration_curves,
                    inputs=[model_choice],
                    outputs=[calib_plot, calib_status]
                )
        
        gr.Markdown("---")
        
        # Inference Section
        gr.Markdown("#### 🔍 Image Prediction")
        
        with gr.Tabs():
            with gr.Tab("Single Image"):
                gr.Markdown("Upload an image to classify it into one of the garbage categories.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        single_image_input = gr.Image(
                            label="Upload Image",
                            type="numpy",
                            height=400
                        )
                        single_track_carbon = gr.Checkbox(
                            label="🌍 Track Carbon Emissions",
                            value=True
                        )
                        single_predict_button = gr.Button("🔍 Predict", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        single_probs_plot = gr.Plot(
                            label="Class Probabilities"
                        )
                
                # Resultado textual debajo
                single_result_text = gr.Markdown("Upload an image and click 'Predict'")
                
                single_predict_button.click(
                    fn=predict_single_image_gradio,
                    inputs=[model_choice, single_image_input, carbon_display, single_track_carbon],
                    outputs=[single_probs_plot, single_result_text, carbon_display]
                )
            
            with gr.Tab("Batch Prediction"):
                gr.Markdown("Upload multiple images to classify them all at once.")
                
                batch_image_input = gr.File(
                    label="Upload Images",
                    file_count="multiple",
                    file_types=["image"]
                )
                batch_track_carbon = gr.Checkbox(
                    label="🌍 Track Carbon Emissions",
                    value=True
                )
                batch_predict_button = gr.Button("🔍 Predict All", variant="primary", size="lg")
                
                batch_result_text = gr.Markdown("Upload images and click 'Predict All'")
                batch_results_table = gr.Dataframe(
                    label="Prediction Results",
                    interactive=False,
                    wrap=True
                )
                
                batch_predict_button.click(
                    fn=predict_folder_gradio,
                    inputs=[model_choice, batch_image_input, carbon_display, batch_track_carbon],
                    outputs=[batch_results_table, batch_result_text, carbon_display]
                )
        
        gr.Markdown("---")
        gr.Markdown(
            "**ℹ️ Info:** Carbon emissions are tracked for inference operations and added to the total carbon footprint."
        )
    
    return []