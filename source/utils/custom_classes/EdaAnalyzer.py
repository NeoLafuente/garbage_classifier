import os
import zipfile
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from typing import Optional


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

    def __init__(self, root_path: str = "../data/raw", dataset_name: str = "Garbage_Dataset_Classification"):
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
    def plot_random_examples_per_class(self, filename: Optional[str] = None):
        """Plot a random image from each class."""
        df = self.df
        classes = df['label'].unique()
        palette = sns.color_palette("tab10", len(classes))
        class_colors = {cls: palette[i] for i, cls in enumerate(classes)}

        cols, rows = 3, (len(classes) + 2) // 3
        plt.figure(figsize=(cols*4, rows*4))

        for i, cls in enumerate(classes):
            img_filename = df[df['label'] == cls].sample(1).iloc[0]['filename']
            img_path = os.path.join(self.dataset_path, "images", cls, img_filename)
            img = Image.open(img_path)

            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(img)
            ax.set_title(cls, fontsize=14, color=class_colors[cls])
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(class_colors[cls])
                spine.set_linewidth(4)

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
        plt.show()

    def plot_class_distribution(self, filename: Optional[str] = None):
        """Plot class distribution using seaborn."""
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.df, x="label", order=self.df['label'].value_counts().index, palette="tab10")
        plt.title("Class Distribution", fontsize=16)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
        plt.show()

    def plot_image_size_scatter(self, filename: Optional[str] = None):
        """Plot scatter of image dimensions per class."""
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
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=size_df, x="Width", y="Height", hue="Label", style="Label", palette="tab10")
        plt.title("Image Dimensions per Class")
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
        plt.show()

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

    def plot_mean_images_per_class(self, filename: Optional[str] = None):
        """Compute and plot mean images per class."""
        mean_images = self._compute_stat_images("mean")
        cols, rows = 3, (len(mean_images) + 2) // 3
        plt.figure(figsize=(cols*4, rows*4))
        for i, (cls, img) in enumerate(mean_images.items()):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(f"Mean {cls}")
            plt.axis("off")
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
        plt.show()
        return mean_images

    def plot_median_images_per_class(self, filename: Optional[str] = None):
        """Compute and plot median images per class."""
        median_images = self._compute_stat_images("median")
        cols, rows = 3, (len(median_images) + 2) // 3
        plt.figure(figsize=(cols*4, rows*4))
        for i, (cls, img) in enumerate(median_images.items()):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(f"Median {cls}")
            plt.axis("off")
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
        plt.show()
        return median_images

    def plot_pixel_distribution_correlation_ordered(self, bins=32):
        """Compute histograms per class, correlate and plot reordered matrix."""
        classes = self.df['label'].unique()
        histograms = {}

        for cls in classes:
            subset = self.df[self.df['label'] == cls]
            hist_accum = None
            for _, row in subset.iterrows():
                img_path = os.path.join(self.dataset_path, "images", row['label'], row['filename'])
                try:
                    img = Image.open(img_path).convert("RGB")
                    arr = np.array(img)
                    hist, _ = np.histogramdd(arr.reshape(-1, 3), bins=(bins, bins, bins), range=((0, 256),)*3)
                    hist_accum = hist if hist_accum is None else hist_accum + hist
                except:
                    continue
            if hist_accum is not None:
                hist_flat = hist_accum.flatten()
                histograms[cls] = hist_flat / np.sum(hist_flat)

        hist_df = pd.DataFrame(histograms).T
        corr = hist_df.T.corr()

        condensed = pdist(corr.values, metric='euclidean')
        link = linkage(condensed, method='average')
        dendro = dendrogram(link, no_plot=True)
        idx = dendro['leaves']
        corr_reordered = corr.values[idx][:, idx]
        reordered_labels = [corr.index[i] for i in idx]

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_reordered, cmap="magma", annot=True, fmt=".2f",
                    xticklabels=reordered_labels, yticklabels=reordered_labels)
        plt.title("Pixel Distribution Correlation (Reordered)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

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
