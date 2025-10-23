import os
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from typing import Optional
import cv2
from source.utils.config import get_valid_dir as gvd


class EdaAnalyzer:
    """
    A class that encapsulates all Exploratory Data Analysis (EDA) utilities
    for the Garbage Classification dataset or similar image datasets.

    This class provides methods for dataset management, visualization,
    and analysis, including downloading datasets from Kaggle,
    loading metadata, plotting class distributions, and computing
    prototype mean images.

    Attributes
    ----------
    root_path : str
        Path to the raw data folder.
    dataset_path : str
        Path to the dataset folder.
    zip_file : str
        Path to the zip file for dataset download.
    kaggle_url : str
        URL for downloading the Kaggle dataset.
    metadata_path : str
        Path to the metadata.csv file.
    df : pd.DataFrame or None
        Metadata DataFrame containing dataset information.
    """

    def __init__(
        self,
        root_path: str = "./data/raw",
        dataset_name: str = "Garbage_Dataset_Classification",
    ):
        """
        Initialize the EdaAnalyzer instance.

        Parameters
        ----------
        root_path : str, optional
            Path to the raw data folder. Default is "./data/raw".
        dataset_name : str, optional
            Name of the dataset folder. Default is
            "Garbage_Dataset_Classification".

        Returns
        -------
        None
        """
        self.root_path = gvd(root_path)
        self.dataset_path = gvd(os.path.join(self.root_path, dataset_name))
        self.zip_file = os.path.join(self.root_path, "garbage-dataset.zip")
        self.dataset_url = (
            "https://github.com/NeoLafuente/garbage_classifier/"
            "releases/download/v0.1.4/garbage-dataset.zip"
        )
        self.metadata_path = os.path.join(self.dataset_path, "metadata.csv")
        self.df = None

    # -------------------------------------------------------------------------
    # Dataset management
    # -------------------------------------------------------------------------
    def download_with_curl(self):
        """
        Download Kaggle dataset using curl and API credentials.

        This method downloads the garbage dataset from Kaggle using the
        Kaggle API credentials stored in ~/.kaggle/kaggle.json. The
        dataset is extracted and the zip file is removed after
        extraction.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If Kaggle credentials are not found at ~/.kaggle/kaggle.json.
        """
        print("Downloading dataset with curl...")
        cmd = f"curl -L -o {self.zip_file} {self.dataset_url}"
        print(cmd)
        os.system(cmd)
        os.chmod(self.zip_file, 0o755)

        print("Extracting dataset...")

        with zipfile.ZipFile(self.zip_file, "r") as zip_ref:
            zip_ref.extractall(self.root_path)

        os.remove(self.zip_file)
        print("Dataset downloaded and extracted successfully.")

    def ensure_dataset(self):
        """
        Check if dataset exists; otherwise, download it.

        Verifies the presence of the dataset at the expected path.
        If not found, triggers the download process.
        If already present, prints a confirmation message.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not os.path.exists(self.metadata_path):
            self.download_with_curl()
        else:
            print(f"{self.dataset_path} already exists, nothing to do.")

    def load_metadata(self):
        """
        Load metadata.csv into a pandas DataFrame.

        Reads the metadata CSV file from the dataset path and stores it as
        self.df. Prints summary statistics about the loaded data.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame (The loaded metadata DataFrame containing image filenames
        and labels).

        Raises
        ------
        FileNotFoundError
            If metadata.csv is not found at the expected path.
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at \
                {self.metadata_path}")
        self.df = pd.read_csv(self.metadata_path)
        print(
            f"Loaded metadata: {len(self.df)} entries, \
                {self.df['label'].nunique()} classes."
        )
        return self.df

    # -------------------------------------------------------------------------
    # Visualization utilities
    # -------------------------------------------------------------------------
    def plot_random_examples_per_class(
        self,
        filename: Optional[str] = None
    ) -> Figure:
        """
        Plot a random image from each class and return the figure.

        Selects one random image per class and displays them in a grid layout.
        Each subplot is bordered with a color corresponding to its class.

        Parameters
        ----------
        filename : str, optional
            Path to save the generated figure as an image file. If provided,
            the figure is saved with 150 dpi. Default is None (no save).

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object containing the plotted images.
        """
        df = self.df
        classes = df["label"].unique()
        palette = sns.color_palette("tab10", len(classes))
        class_colors = {cls: palette[i] for i, cls in enumerate(classes)}

        cols, rows = 3, (len(classes) + 2) // 3
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten()

        for i, cls in enumerate(classes):
            img_filename = df[df["label"] == cls].sample(1).iloc[0]["filename"]
            img_path = os.path.join(
                self.dataset_path,
                "images",
                cls,
                img_filename
            )
            img = Image.open(img_path)

            ax = axes[i]
            ax.imshow(img)
            ax.set_title(cls, fontsize=14, color=class_colors[cls])
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(class_colors[cls])
                spine.set_linewidth(4)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=150)

        return fig

    def plot_class_distribution(
        self,
        filename: Optional[str] = None
    ) -> Figure:
        """
        Plot class distribution using seaborn and return the figure.

        Creates a countplot showing the number of samples per class, ordered
        by frequency and using a color palette for visual distinction.

        Parameters
        ----------
        filename : str, optional
            Path to save the generated figure as an image file. If provided,
            the figure is saved with 150 dpi. Default is None (no save).

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object containing the class distribution plot.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            data=self.df,
            x="label",
            order=self.df["label"].value_counts().index,
            palette="tab10",
            ax=ax,
        )
        ax.set_title("Class Distribution", fontsize=16)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()

        if filename:
            fig.savefig(filename, dpi=150)

        return fig

    # -------------------------------------------------------------------------
    # Prototypes
    # -------------------------------------------------------------------------

    def _compute_mean_images_per_batch(self, batch_size=32):
        """
        Compute mean image per class using batch processing.

        Processes images in batches to compute the mean image for each class,
        reducing memory overhead for large datasets.
        Images are converted to RGB and normalized to float32.

        Parameters
        ----------
        batch_size : int, optional
            Number of images to process per batch. Default is 32.

        Returns
        -------
        dict
            Dictionary with class names as keys and normalized mean images
            (values in range [0, 1]) as values.

        Notes
        -----
        Images are normalized by dividing by 255.0. Invalid or corrupted images
        are skipped during processing.
        """

        classes = self.df["label"].unique()
        result = {}

        for cls in classes:
            subset = self.df[self.df["label"] == cls]
            count = 0
            mean_acc = None

            for batch_start in range(0, len(subset), batch_size):
                batch_end = min(batch_start + batch_size, len(subset))
                batch_rows = subset.iloc[batch_start:batch_end]

                imgs = []
                for _, row in batch_rows.iterrows():
                    img_path = os.path.join(
                        self.dataset_path,
                        "images",
                        row["label"],
                        row["filename"]
                    )
                    try:
                        img = Image.open(img_path).convert("RGB")
                        imgs.append(np.array(img, dtype=np.float32))
                    except Exception:
                        continue

                if imgs:
                    imgs_stack = np.stack(imgs, axis=0)
                    batch_mean = np.mean(imgs_stack, axis=0)

                    # Actualizar media acumulada
                    if mean_acc is None:
                        mean_acc = batch_mean
                    else:
                        aux1 = mean_acc * count
                        aux2 = batch_mean * len(imgs)
                        mean_acc = (aux1 + aux2) / (
                            count + len(imgs)
                        )

                    count += len(imgs)

            if mean_acc is not None:
                result[cls] = mean_acc / 255.0

        return result

    def plot_mean_images_per_class(
        self,
        filename: Optional[str] = None
    ) -> Figure:
        """
        Compute or load and plot mean images per class, returning the figure.

        Attempts to load pre-computed mean images from a .npy file.
        If not found, computes them using batch processing and
        optionally saves the result.
        Displays all mean images in a grid layout.

        Parameters
        ----------
        filename : str, optional
            Path to the .npy file containing pre-computed mean images,
            or destination path for saving newly computed mean images.
            Default is None (no caching).

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object containing the plotted mean images.

        Notes
        -----
        If filename is provided and the file does not exist, computed
        mean images will be saved to this path for future use.
        """

        mean_images = None

        if filename and os.path.exists(filename):
            try:
                print(f"[INFO] Loading mean images from {filename}")
                mean_images = np.load(filename, allow_pickle=True).item()
            except Exception as e:
                print(f"[WARN] Could not load \
                    mean images from {filename}: {e}")

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

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        return fig

    def plot_mean_images_per_class_with_otsu(
        self, threshold: float = 0.0, filename: Optional[str] = None
    ) -> Figure:
        """
        Plot mean images per class applying an adjustable Otsu threshold.

        Loads pre-computed mean images and applies a custom thresholding
        strategy based on Otsu's method with user-defined adjustments.
        Generates binary masks and overlays them on the original mean
        images with contour visualization.

        Parameters
        ----------
        threshold : float, optional
            Threshold adjustment parameter. Range: [-1, 1].
            - -1: Maximum threshold (255, minimal foreground).
            - 0: Otsu threshold (default).
            - 1: Minimum threshold (0, maximal foreground).
            Default is 0.0.
        filename : str, optional
            Path to the .npy file containing pre-computed mean images.
            Must end with ".npy" extension. Default is None.

        Returns
        -------
        matplotlib.figure.Figure or None
            The generated figure object containing the thresholded mean images.
            Returns None if mean images cannot be loaded or invalid parameters
            are provided.

        Raises
        ------
        None

        Notes
        -----
        Red overlays indicate pixels below the threshold (potential
        foreground objects). Contours are traced around connected
        components in the binary mask.
        """

        mean_images = None

        if filename and os.path.exists(filename) and filename.endswith(".npy"):
            try:
                print(f"[INFO] Loading mean images from {filename}")
                mean_images = np.load(filename, allow_pickle=True).item()
            except Exception as e:
                print(f"[WARN] Could not load \
                    mean images from {filename}: {e}")
                return None
        else:
            print("[WARN] No mean images found or invalid file path.")
            return None

        n_classes = len(mean_images)
        n_cols = min(3, n_classes)
        n_rows = int(np.ceil(n_classes / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5 * n_cols, 5 * n_rows)
        )
        axes = np.array(axes).flatten()

        for i, (cls, mean_image) in enumerate(mean_images.items()):
            ax = axes[i]
            gray = cv2.cvtColor(mean_image, cv2.COLOR_RGB2GRAY)

            if gray.dtype != np.uint8:
                gray = cv2.normalize(
                    gray, None, 0, 255, cv2.NORM_MINMAX
                ).astype(
                    np.uint8
                )

            otsu_thresh, _ = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

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

            _, binary = cv2.threshold(
                gray,
                final_thresh,
                255,
                cv2.THRESH_BINARY
            )

            mask = (binary == 0).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            red_overlay = np.zeros((*mask.shape, 4))
            red_overlay[mask_dilated == 1] = [1, 0, 0, 0.25]

            ax.imshow(mean_image)
            ax.imshow(red_overlay)
            ax.set_title(f"{cls}\nOtsu adj={threshold:.2f} \
                (thr={final_thresh:.1f})")
            ax.axis("off")

            contours, _ = cv2.findContours(
                mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                contour = contour.squeeze()
                if contour.ndim == 2:
                    ax.plot(
                        contour[:, 0],
                        contour[:, 1],
                        color="red",
                        linewidth=2
                    )

        plt.tight_layout()

        return fig
