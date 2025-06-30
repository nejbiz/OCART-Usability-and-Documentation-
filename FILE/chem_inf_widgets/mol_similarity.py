from Orange.widgets import widget, gui, settings
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.metrics.pairwise import cosine_similarity

# Background worker for similarity calculation.
class SimilarityWorker(QThread):
    progress = pyqtSignal(int)       # Emits current progress (0-100)
    finished = pyqtSignal(np.ndarray)  # Emits the computed similarity matrix

    def __init__(self, fps, metric, alpha=0.5, beta=0.5, parent=None):
        super().__init__(parent)
        self.fps = fps
        self.metric = metric
        self.alpha = alpha
        self.beta = beta

    def run(self):
        print(f"[DEBUG] Starting similarity calculation using metric: {self.metric}")
        n = self.fps.shape[0]
        sim_matrix = np.zeros((n, n))
        if self.metric in ["Tanimoto", "Dice", "Tversky"]:
            # Compute the intersection (dot product) for all pairs.
            intersection = np.dot(self.fps, self.fps.T)
            print(f"[DEBUG] Intersection computed, shape: {intersection.shape}")
            # Compute the number of "on" bits for each fingerprint.
            sums = np.sum(self.fps, axis=1)
            print(f"[DEBUG] Fingerprint bit sums: {sums}")
            if self.metric == "Tanimoto":
                denom = np.add.outer(sums, sums) - intersection
                print("[DEBUG] Tanimoto: Denominator calculated.")
                with np.errstate(divide="ignore", invalid="ignore"):
                    sim_matrix = np.true_divide(intersection, denom)
                    sim_matrix[denom == 0] = 0.0
            elif self.metric == "Dice":
                denom = np.add.outer(sums, sums)
                print("[DEBUG] Dice: Denominator calculated.")
                with np.errstate(divide="ignore", invalid="ignore"):
                    sim_matrix = 2 * intersection / denom
                    sim_matrix[denom == 0] = 0.0
            elif self.metric == "Tversky":
                A = sums.reshape(-1, 1)
                B = sums.reshape(1, -1)
                print(f"[DEBUG] Tversky: Calculating denominator with alpha={self.alpha} and beta={self.beta}")
                denom = intersection + self.alpha * (A - intersection) + self.beta * (B - intersection)
                with np.errstate(divide="ignore", invalid="ignore"):
                    sim_matrix = np.true_divide(intersection, denom)
                    sim_matrix[denom == 0] = 0.0
        elif self.metric == "Cosine":
            print("[DEBUG] Calculating Cosine similarity.")
            sim_matrix = cosine_similarity(self.fps)
        else:
            print("[DEBUG] Unknown similarity metric selected. Returning empty matrix.")
            sim_matrix = np.array([])

        print(f"[DEBUG] Similarity matrix computed, shape: {sim_matrix.shape}")
        self.progress.emit(100)
        self.finished.emit(sim_matrix)

class MolecularSimilarityWidget(widget.OWWidget):
    name = "Molecular Similarity Calculator"
    description = "Calculates a molecular similarity matrix from fingerprint data using various similarity metrics."
    icon = "icons/similarity.png"
    priority = 24

    inputs = [("Fingerprints", Table, "set_fingerprints")]
    outputs = [("Similarity Matrix", Table)]

    # User settings.
    similarity_metric = settings.Setting("Tanimoto")
    tversky_alpha = settings.Setting(0.5)
    tversky_beta = settings.Setting(0.5)

    def __init__(self):
        super().__init__()
        self.fingerprint_table = None
        self.similarity_matrix = None
        self.worker = None

        # Similarity Settings UI.
        box = gui.widgetBox(self.controlArea, "Similarity Settings")
        self.metric_combo = gui.comboBox(
            box, self, "similarity_metric",
            label="Similarity Metric:",
            items=["Tanimoto", "Cosine", "Dice", "Tversky"],
            orientation=gui.Qt.Horizontal
        )
        # Use doubleSpin for floating-point parameters.
        self.alpha_spin = gui.doubleSpin(
            box, self, "tversky_alpha",
            minv=0.0, maxv=1.0, step=0.1,
            label="Tversky Alpha",
            controlWidth=60
        )
        self.beta_spin = gui.doubleSpin(
            box, self, "tversky_beta",
            minv=0.0, maxv=1.0, step=0.1,
            label="Tversky Beta",
            controlWidth=60
        )
        gui.button(box, self, "Compute Similarity", callback=self.compute_similarity)
        gui.button(box, self, "Show Heatmap", callback=self.show_heatmap)

    def set_fingerprints(self, data):
        """
        Receives the fingerprint table from a Fingerprint Calculator widget.
        The table is expected to have fingerprint bits as attributes and a meta column (e.g., 'SMILES').
        """
        self.fingerprint_table = data
        self.similarity_matrix = None
        self.info(f"[DEBUG] Received fingerprint table: {data.X.shape[0]} rows, {data.X.shape[1]} columns.")

    def compute_similarity(self):
        """Starts the background thread for similarity matrix calculation."""
        if self.fingerprint_table is None:
            self.error("No fingerprint data received!")
            return

        fingerprint_data = np.array(self.fingerprint_table.X)
        self.info(f"[DEBUG] Fingerprint data shape: {fingerprint_data.shape}")
        self.progressBarInit(100)
        metric = self.similarity_metric
        if metric == "Tversky":
            alpha = self.tversky_alpha
            beta = self.tversky_beta
            self.info(f"[DEBUG] Using Tversky parameters: alpha={alpha}, beta={beta}")
        else:
            alpha = 0.5
            beta = 0.5

        self.worker = SimilarityWorker(fingerprint_data, metric, alpha, beta)
        self.worker.progress.connect(self.progressBarSet)
        self.worker.finished.connect(self.handle_similarity_finished)
        self.worker.start()

    def handle_similarity_finished(self, sim_matrix):
        """Processes the computed similarity matrix and sends the output."""
        if sim_matrix.ndim != 2 or sim_matrix.size == 0:
            self.error("Computed similarity matrix is empty or not two-dimensional!")
            return

        self.similarity_matrix = sim_matrix
        self.info(f"[DEBUG] Similarity matrix shape: {self.similarity_matrix.shape}")
        self.progressBarFinished()

        n_samples = self.similarity_matrix.shape[0]
        if n_samples == 0:
            self.error("No molecules available in the similarity matrix!")
            return

        if self.fingerprint_table.domain.metas and len(self.fingerprint_table.domain.metas) > 0:
            mol_labels = [str(row[0]) for row in self.fingerprint_table.metas]
        else:
            mol_labels = [f"Molecule {i}" for i in range(n_samples)]
        attributes = [ContinuousVariable(label) for label in mol_labels]
        domain = Domain(attributes)
        similarity_table = Table(domain, self.similarity_matrix)
        self.send("Similarity Matrix", similarity_table)
        self.info(f"[DEBUG] Similarity table sent with shape: {similarity_table.X.shape}")

    def show_heatmap(self):
        """Displays a heatmap of the computed similarity matrix."""
        if self.similarity_matrix is None or self.similarity_matrix.size == 0:
            self.error("Similarity matrix has not been computed yet or is empty!")
            return
        plt.figure(figsize=(8, 6))
        plt.imshow(self.similarity_matrix, interpolation="nearest", cmap="viridis")
        plt.colorbar()
        plt.title(f"Similarity Matrix ({self.similarity_metric})")
        plt.xlabel("Molecules")
        plt.ylabel("Molecules")
        plt.show()
