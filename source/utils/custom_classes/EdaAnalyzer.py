import os
import zipfile
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from typing import Optional
import cv2


class EdaAnalyzer:
    """
    A class that encapsulates all Exploratory Data Analysis (EDA) utilities
    for the Garbage Classification dataset (or similar image datasets).

    Attributes
    ----------
    root_path : str
        Path to the raw data folder
    dataset_path : str
        Path to the dataset folder
    metadata_path : str
        Path to the metadata.csv file
    df : pd.DataFrame
        Metadata DataFrame
    """

    def __init__(self, root_path: str = "./data/raw", dataset_name: str = "Garbage_Dataset_Classification"):
        self.root_path = root_path
        self.dataset_path = os.path.join(root_path, dataset_name)
        self.zip_file = os.path.join(root_path, "garbage-dataset.zip")
        self.kaggle_url = "https://www.kaggle.com/api/v1/datasets/download/zlatan599/garbage-dataset-classification"
        self.metadata_path = os.path.join(self.dataset_path, "metadata.csv")
        self.df = None

    # -------------------------------------------------------------------------
    # Dataset management
    # -------------------------------------------------------------------------
    def download_with_curl(self):
        """Download Kaggle dataset using curl + API credentials."""
        print("Downloading dataset with curl...")

        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        os.chmod(os.path.expanduser("~/.kaggle"), 0o700)

        cmd = f"curl -L -o {self.zip_file} -u `jq -r .username ~/.kaggle/kaggle.json`:`jq -r .key ~/.kaggle/kaggle.json` {self.kaggle_url}"
        os.system(cmd)

        print("Extracting dataset...")
        with zipfile.ZipFile(self.zip_file, "r") as zip_ref:
            zip_ref.extractall(self.root_path)

        os.remove(self.zip_file)
        print("Dataset downloaded and extracted successfully.")

    def ensure_dataset(self):
        """Check if dataset exists; otherwise, download it."""
        if not os.path.exists(self.dataset_path):
            self.download_with_curl()
        else:
            print(f"{self.dataset_path} already exists, nothing to do.")

    def load_metadata(self):
        """Load metadata.csv into a pandas DataFrame."""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
        self.df = pd.read_csv(self.metadata_path)
        print(f"Loaded metadata: {len(self.df)} entries, {self.df['label'].nunique()} classes.")
        return self.df

    # -------------------------------------------------------------------------
    # Visualization utilities
    # -------------------------------------------------------------------------
    def plot_random_examples_per_class(self, filename: Optional[str] = None) -> Figure:
        """Plot a random image from each class and return the figure."""
        df = self.df
        classes = df['label'].unique()
        palette = sns.color_palette("tab10", len(classes))
        class_colors = {cls: palette[i] for i, cls in enumerate(classes)}

        cols, rows = 3, (len(classes) + 2) // 3
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        axes = axes.flatten() 

        for i, cls in enumerate(classes):
            img_filename = df[df['label'] == cls].sample(1).iloc[0]['filename']
            img_path = os.path.join(self.dataset_path, "images", cls, img_filename)
            img = Image.open(img_path)

            ax = axes[i]
            ax.imshow(img)
            ax.set_title(cls, fontsize=14, color=class_colors[cls])
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(class_colors[cls])
                spine.set_linewidth(4)

        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=150)
        
        return fig

    def plot_class_distribution(self, filename: Optional[str] = None) -> Figure:
        """Plot class distribution using seaborn and return the figure."""
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            data=self.df,
            x="label",
            order=self.df['label'].value_counts().index,
            palette="tab10",
            ax=ax
        )
        ax.set_title("Class Distribution", fontsize=16)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()

        if filename:
            fig.savefig(filename, dpi=150)
        
        return fig

    def plot_image_size_scatter(self, filename: Optional[str] = None) -> Figure:
        """Plot scatter of image dimensions per class and return the figure."""
        widths, heights, labels = [], [], []
        for _, row in self.df.iterrows():
            img_path = os.path.join(self.dataset_path, "images", row['label'], row['filename'])
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                widths.append(w)
                heights.append(h)
                labels.append(row['label'])
            except:
                continue

        size_df = pd.DataFrame({"Width": widths, "Height": heights, "Label": labels})
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=size_df,
            x="Width",
            y="Height",
            hue="Label",
            style="Label",
            palette="tab10",
            ax=ax
        )
        ax.set_title("Image Dimensions per Class")
        plt.tight_layout()

        if filename:
            fig.savefig(filename, dpi=150)

        return fig

    # -------------------------------------------------------------------------
    # Prototypes & correlations
    # -------------------------------------------------------------------------
    def _compute_stat_images(self, statistic="mean"):
        """Helper: compute mean or median image per class."""
        classes = self.df['label'].unique()
        result = {}
        for cls in classes:
            imgs = []
            subset = self.df[self.df['label'] == cls]
            for _, row in subset.iterrows():
                img_path = os.path.join(self.dataset_path, "images", row['label'], row['filename'])
                try:
                    img = Image.open(img_path).convert("RGB")
                    imgs.append(np.array(img, dtype=np.float32))
                except:
                    continue
            if imgs:
                imgs_stack = np.stack(imgs, axis=0)
                if statistic == "mean":
                    result[cls] = np.mean(imgs_stack, axis=0) / 255.0
                else:
                    result[cls] = np.median(imgs_stack, axis=0) / 255.0
        return result

    def _compute_mean_images_per_batch(self, batch_size=32):
        """Helper: compute mean image per class using batch processing."""
        classes = self.df['label'].unique()
        result = {}
        
        for cls in classes:
            subset = self.df[self.df['label'] == cls]
            count = 0
            mean_acc = None
            
            for batch_start in range(0, len(subset), batch_size):
                batch_end = min(batch_start + batch_size, len(subset))
                batch_rows = subset.iloc[batch_start:batch_end]
                
                imgs = []
                for _, row in batch_rows.iterrows():
                    img_path = os.path.join(self.dataset_path, "images", row['label'], row['filename'])
                    try:
                        img = Image.open(img_path).convert("RGB")
                        imgs.append(np.array(img, dtype=np.float32))
                    except:
                        continue
                
                if imgs:
                    imgs_stack = np.stack(imgs, axis=0)
                    batch_mean = np.mean(imgs_stack, axis=0)
                    
                    # Actualizar media acumulada
                    if mean_acc is None:
                        mean_acc = batch_mean
                    else:
                        mean_acc = (mean_acc * count + batch_mean * len(imgs)) / (count + len(imgs))
                    
                    count += len(imgs)
            
            if mean_acc is not None:
                result[cls] = mean_acc / 255.0
        
        return result

    def plot_mean_images_per_class(self, filename: Optional[str] = None) -> Figure:
        """Compute or load and plot mean images per class, returning the figure."""
        
        mean_images = None

        if filename and os.path.exists(filename):
            try:
                print(f"[INFO] Loading mean images from {filename}")
                mean_images = np.load(filename, allow_pickle=True).item()
            except Exception as e:
                print(f"[WARN] Could not load mean images from {filename}: {e}")

        if mean_images is None:
            print("[INFO] Computing mean images...")
            mean_images = self._compute_mean_images_per_batch()
            if filename:
                np.save(filename, mean_images)
                print(f"[INFO] Saved mean images to {filename}")

        # --- Plot ---
        cols, rows = 3, (len(mean_images) + 2) // 3
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten() 

        for i, (cls, img) in enumerate(mean_images.items()):
            ax = axes[i]
            ax.imshow(img)
            ax.set_title(f"Mean {cls}")
            ax.axis("off")

        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        return fig

    def plot_mean_images_per_class_with_otsu(self, threshold: float = 0.0, filename: Optional[str] = None) -> Figure:
        """
        Plots the mean images per class applying an adjustable Otsu threshold.

        Parameters:
            threshold (float): Threshold adjustment (-1 = maximum, 0 = Otsu, 1 = minimum)
            filename (str, optional): Path to the .npy file containing mean_images, 
                                      or destination PDF path if saving the figure.

        Returns:
            fig (matplotlib.figure.Figure): Generated figure.
        """

        mean_images = None

        if filename and os.path.exists(filename) and filename.endswith(".npy"):
            try:
                print(f"[INFO] Loading mean images from {filename}")
                mean_images = np.load(filename, allow_pickle=True).item()
            except Exception as e:
                print(f"[WARN] Could not load mean images from {filename}: {e}")
                return None
        else:
            print("[WARN] No mean images found or invalid file path.")
            return None

        n_classes = len(mean_images)
        n_cols = min(3, n_classes)
        n_rows = int(np.ceil(n_classes / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = np.array(axes).flatten()

        for i, (cls, mean_image) in enumerate(mean_images.items()):
            ax = axes[i]
            gray = cv2.cvtColor(mean_image, cv2.COLOR_RGB2GRAY)

            if gray.dtype != np.uint8:
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            adj = np.clip(threshold, -1, 1)
            if adj == -1:
                final_thresh = 255
            elif adj == 1:
                final_thresh = 0
            else:
                if adj < 0:
                    final_thresh = otsu_thresh + (255 - otsu_thresh) * (-adj)
                else:
                    final_thresh = otsu_thresh - (otsu_thresh - 0) * adj

            _, binary = cv2.threshold(gray, final_thresh, 255, cv2.THRESH_BINARY)

            mask = (binary == 0).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            red_overlay = np.zeros((*mask.shape, 4))
            red_overlay[mask_dilated == 1] = [1, 0, 0, 0.25]

            ax.imshow(mean_image)
            ax.imshow(red_overlay)
            ax.set_title(f"{cls}\nOtsu adj={threshold:.2f} (thr={final_thresh:.1f})")
            ax.axis("off")

            contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.squeeze()
                if contour.ndim == 2:
                    ax.plot(contour[:, 0], contour[:, 1], color="red", linewidth=2)

        plt.tight_layout()

        return fig

    def plot_median_images_per_class(self, filename: Optional[str] = None) -> Figure:
        """Compute or load and plot median images per class, returning the figure."""
        
        median_images = None

        if filename and os.path.exists(filename):
            try:
                print(f"[INFO] Loading median images from {filename}")
                median_images = np.load(filename, allow_pickle=True).item()
            except Exception as e:
                print(f"[WARN] Could not load median images from {filename}: {e}")

        if median_images is None:
            print("[INFO] Computing median images...")
            median_images = self._compute_stat_images("median")
            if filename:
                np.save(filename, median_images)
                print(f"[INFO] Saved median images to {filename}")

        # --- Plot ---
        cols, rows = 3, (len(median_images) + 2) // 3
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten()

        for i, (cls, img) in enumerate(median_images.items()):
            ax = axes[i]
            ax.imshow(img)
            ax.set_title(f"Median {cls}")
            ax.axis("off")

        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        return fig

    def plot_median_images_per_class_with_otsu(self, threshold: float = 0.0, filename: Optional[str] = None) -> Figure:
        """
        Plots the median images per class applying an adjustable Otsu threshold.
        Parameters:
            threshold (float): Threshold adjustment (-1 = maximum, 0 = Otsu, 1 = minimum)
            filename (str, optional): Path to the .npy file containing median_images, 
                                    or destination PDF path if saving the figure.
        Returns:
            fig (matplotlib.figure.Figure): Generated figure.
        """
        median_images = None
        
        if filename and os.path.exists(filename) and filename.endswith(".npy"):
            try:
                print(f"[INFO] Loading median images from {filename}")
                median_images = np.load(filename, allow_pickle=True).item()
            except Exception as e:
                print(f"[WARN] Could not load median images from {filename}: {e}")
        
        if median_images is None:
            print("[INFO] Computing median images...")
            median_images = self._compute_stat_images("median")
            if filename and filename.endswith(".npy"):
                np.save(filename, median_images)
                print(f"[INFO] Saved median images to {filename}")
        
        n_classes = len(median_images)
        n_cols = min(3, n_classes)
        n_rows = int(np.ceil(n_classes / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = np.array(axes).flatten()
        
        for i, (cls, median_image) in enumerate(median_images.items()):
            ax = axes[i]
            gray = cv2.cvtColor(median_image, cv2.COLOR_RGB2GRAY)
            if gray.dtype != np.uint8:
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adj = np.clip(threshold, -1, 1)
            if adj == -1:
                final_thresh = 255
            elif adj == 1:
                final_thresh = 0
            else:
                if adj < 0:
                    final_thresh = otsu_thresh + (255 - otsu_thresh) * (-adj)
                else:
                    final_thresh = otsu_thresh - (otsu_thresh - 0) * adj
            _, binary = cv2.threshold(gray, final_thresh, 255, cv2.THRESH_BINARY)
            mask = (binary == 0).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            red_overlay = np.zeros((*mask.shape, 4))
            red_overlay[mask_dilated == 1] = [1, 0, 0, 0.25]
            ax.imshow(median_image)
            ax.imshow(red_overlay)
            ax.set_title(f"{cls}\nOtsu adj={threshold:.2f} (thr={final_thresh:.1f})")
            ax.axis("off")
            contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.squeeze()
                if contour.ndim == 2:
                    ax.plot(contour[:, 0], contour[:, 1], color="red", linewidth=2)
        
        plt.tight_layout()
        return fig

    def compute_cosine_similarity(self, mean_images: dict, channel: Optional[int] = None):
        """
        Compute cosine similarity between mean images (optionally by channel).
        """
        if channel is not None:
            X = np.array([img[:, :, channel].flatten() for img in mean_images.values()])
        else:
            X = np.array([img.flatten() for img in mean_images.values()])

        cos_sim = cosine_similarity(X)
        return pd.DataFrame(cos_sim, index=mean_images.keys(), columns=mean_images.keys())
