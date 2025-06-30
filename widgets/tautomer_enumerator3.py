import os
import subprocess
import re
import shutil
import numpy as np
import tempfile
import io
import base64
import random  # For random number generation
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize

from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets import gui
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from AnyQt.QtWidgets import (
    QRadioButton, QCheckBox, QLabel, QSpinBox, QTextEdit, QFileDialog, QComboBox
)
from AnyQt.QtCore import Qt, QThread, pyqtSignal

# For PDF generation (requires xhtml2pdf package)
from xhtml2pdf import pisa

# ------------------------------------------------------------
# Helper function to clear lingering temporary files and cache
# ------------------------------------------------------------
def clear_temp_files_and_cache():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xtb_data_dir = os.path.join(script_dir, "xtb_data")
    if os.path.exists(xtb_data_dir):
        for item in os.listdir(xtb_data_dir):
            path = os.path.join(xtb_data_dir, item)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception as e:
                print(f"Could not remove {path}: {e}")

# ------------------------------------------------------------
# Utility: search for "SMILES" in an Orange table (all rows)
# ------------------------------------------------------------
def get_smiles_list_from_table(table):
    domain = getattr(table, "domain", None)
    if domain is None:
        return []
    all_vars = list(domain.variables) + list(domain.class_vars) + list(domain.metas)
    smiles_var = None
    for var in all_vars:
        if var.name.lower() == "smiles":
            smiles_var = var
            break
    if smiles_var is None:
        return []
    col_data = table.get_column(smiles_var)
    return [str(x) for x in col_data if x is not None and str(x).strip()]

# ------------------------------------------------------------
# RDKit / xTB functions
# ------------------------------------------------------------
def enumerate_tautomers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES provided!")
    enumerator = rdMolStandardize.TautomerEnumerator()
    tautomers = enumerator.Enumerate(mol)
    return [Chem.MolToSmiles(t) for t in tautomers] if tautomers else [smiles]

def generate_3d_structure(smiles, method="rdkit_conversion", calc_dir=None):
    """
    Generate a 3D structure for the given SMILES.
    For "openbabel_conformer_generation", OpenBabel is used to generate 100 conformers
    and select the lowest-energy conformer; otherwise, RDKit is used.
    """
    if method == "rdkit_conversion":
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        return mol
    elif method == "uff_minimisation":
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        ff = AllChem.UFFGetMoleculeForceField(mol)
        ff.Minimize()
        return mol
    elif method == "openbabel_conformer_generation":
        if calc_dir is None:
            calc_dir = tempfile.mkdtemp()
        return generate_3d_structure_openbabel(smiles, nconf=100, workdir=calc_dir)
    else:
        raise ValueError("Unknown method for 3D structure generation")

def generate_3d_structure_openbabel(smiles, nconf=100, workdir=None):
    """
    Use OpenBabel to generate 3D coordinates by creating conformers and selecting the lowest-energy one.
    """
    if workdir is None:
        workdir = tempfile.mkdtemp()
    input_file = os.path.join(workdir, "input.smi")
    output_file = os.path.join(workdir, "output.sdf")
    
    with open(input_file, "w") as f:
        f.write(f"{smiles}\t1\n")
    
    cmd = [
        "obabel",
        "-ismi", input_file,
        "-O", output_file,
        "--gen3d",
        "--conformer",
        f"--nconf", str(nconf),
        "--score"
    ]
    
    try:
        subprocess.check_call(cmd, cwd=workdir)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"OpenBabel conformer generation failed: {e}")
    
    suppl = Chem.SDMolSupplier(output_file, removeHs=False)
    best_mol = None
    best_energy = None
    for mol in suppl:
        if mol is None:
            continue
        try:
            energy = float(mol.GetProp("Energy"))
        except (KeyError, ValueError):
            energy = None
        if energy is not None:
            if best_energy is None or energy < best_energy:
                best_energy = energy
                best_mol = mol
        else:
            if best_mol is None:
                best_mol = mol
    if best_mol is None:
        raise RuntimeError("No valid conformers generated by OpenBabel.")
    return best_mol

def calculate_uff_energy(mol):
    ff = AllChem.UFFGetMoleculeForceField(mol)
    return ff.CalcEnergy()

def write_xyz_file(mol, filename):
    conf = mol.GetConformer()
    with open(filename, 'w') as f:
        f.write(f"{mol.GetNumAtoms()}\n\n")
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

def run_xtb(filename, workdir, optimize=True, solvent="water", thermo=True):
    cmd = ['xtb', filename]
    if optimize:
        cmd.append('--opt')
    if solvent:
        cmd.extend(['--alpb', solvent])
    if thermo:
        cmd.append('--hess')
    
    log_file = os.path.join(workdir, 'xtb_output.log')
    with open(log_file, 'w') as log:
        process = subprocess.Popen(cmd, cwd=workdir, stdout=log, stderr=log)
        process.wait()
    return log_file

def extract_total_energy(log_file):
    pattern = r'\|\s*TOTAL ENERGY\s+([-+]?\d*\.\d+)\s*Eh'
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                return float(match.group(1))
    raise ValueError("Total energy not found in xTB output.")

# ------------------------------------------------------------
# Create output table with proper cleanup handling
# ------------------------------------------------------------
def create_output_table(
    original_smiles_list,
    all_tautomer_smiles,
    rdkit_scores,
    rank_rdkit_list,
    rank_xtb_list,
    rel_energy_list,
    probabilities_list,
    tautomer_counts,
    threshold_counts,
    tautomer_scores_list,
    deltaG_threshold  # Note: deltaG_threshold is still used internally
):
    domain = Domain(
        [
            ContinuousVariable("RDKIT_SCORE"),
            ContinuousVariable("RDKIT_RANK"),
            ContinuousVariable("XTB_RANK"),
            ContinuousVariable("XTB_DG"),
            ContinuousVariable("XTB_PROB"),
            ContinuousVariable("TAUTOMER_COUNT"),
            ContinuousVariable("TAUTOMER_COUNT_TRESHOLD"),
            ContinuousVariable("TAUTOMER_SCORE")
        ],
        metas=[
            StringVariable("ORIG_SMILES"),
            StringVariable("SMILES")
        ]
    )

    rows = []
    for mol_idx, orig_smiles in enumerate(original_smiles_list):
        n = len(all_tautomer_smiles[mol_idx])
        rdkit_rank_of = [0] * n
        for rank_i, taut_idx in enumerate(rank_rdkit_list[mol_idx]):
            rdkit_rank_of[taut_idx] = rank_i + 1
        xtb_rank_of = [0] * n
        for rank_i, taut_idx in enumerate(rank_xtb_list[mol_idx]):
            xtb_rank_of[taut_idx] = rank_i + 1

        for i in range(n):
            # Only include tautomers that fall below the threshold.
            if rel_energy_list[mol_idx][i] <= deltaG_threshold:
                rows.append([
                    float(rdkit_scores[mol_idx]),
                    float(rdkit_rank_of[i]),
                    float(xtb_rank_of[i]),
                    float(rel_energy_list[mol_idx][i]),
                    float(round(probabilities_list[mol_idx][i] * 100, 2)),
                    float(tautomer_counts[mol_idx]),
                    float(threshold_counts[mol_idx]),
                    float(tautomer_scores_list[mol_idx][i]),
                    str(orig_smiles),
                    str(all_tautomer_smiles[mol_idx][i])
                ])

    if not rows:
        data = np.zeros((0, 8), dtype=float)
        metas = np.zeros((0, 2), dtype=object)
    else:
        data = np.array([row[:8] for row in rows], dtype=float)
        metas = np.array([[row[8], row[9]] for row in rows], dtype=object)

    return Table.from_numpy(
        domain=domain,
        X=data,
        metas=metas
    )

# ------------------------------------------------------------
# Background worker with directory management and optional debug info
# ------------------------------------------------------------
class MultiTautomerWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)

    # A default ΔG threshold is now set (e.g. 100 kcal/mol)
    DEFAULT_DG_THRESHOLD = 100

    def __init__(self, smiles_list, method, optimize, solvent, thermo,
                 max_tautomers, remove_temp=False, debug=False):
        super().__init__()
        self.smiles_list = smiles_list
        self.method = method
        self.optimize = optimize
        self.solvent = solvent
        self.thermo = thermo
        # Use default threshold since the slider is removed.
        self.deltaG_threshold = self.DEFAULT_DG_THRESHOLD  
        self.max_tautomers = max_tautomers
        self.remove_temp = remove_temp
        self.debug = debug

    def run(self):
        overall_report = []  # Only used if debug is True
        original_smiles_list = []
        all_tautomer_smiles = []
        rdkit_scores = []
        rank_rdkit_list = []
        rank_xtb_list = []
        rel_energy_list = []
        probabilities_list = []
        tautomer_counts = []
        threshold_counts = []
        tautomer_scores_list = []
        
        def log(msg):
            if self.debug:
                overall_report.append(msg)
        
        temp_dirs = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xtb_data_dir = os.path.join(script_dir, "xtb_data")
        os.makedirs(xtb_data_dir, exist_ok=True)

        for mol_idx, smiles in enumerate(self.smiles_list):
            safe_smiles = re.sub(r'\W+', '', smiles)
            random_number = random.randint(1000, 9999)
            base_dir = os.path.join(xtb_data_dir, f"tautomer_{mol_idx}_{safe_smiles}_{random_number}")
            os.makedirs(base_dir, exist_ok=True)
            temp_dirs.append(base_dir)
            log(f"Created temporary directory for molecule {mol_idx+1}: {base_dir}")
            
            try:
                self.process_molecule(
                    mol_idx, smiles, base_dir, overall_report,
                    original_smiles_list, all_tautomer_smiles,
                    rdkit_scores, rank_rdkit_list, rank_xtb_list,
                    rel_energy_list, probabilities_list,
                    tautomer_counts, threshold_counts, tautomer_scores_list
                )
            except Exception as e:
                if self.debug:
                    overall_report.append(f"Error processing molecule {smiles}: {str(e)}")
            finally:
                log(f"Temporary directory kept for molecule {mol_idx+1}: {base_dir}")
        
        if self.debug:
            log("\nSummary of temporary directories:")
            for d in temp_dirs:
                log(d)
        
        # Remove temporary directories if requested.
        if self.remove_temp:
            for d in temp_dirs:
                try:
                    shutil.rmtree(d)
                    if self.debug:
                        log(f"Removed temporary directory: {d}")
                except Exception as e:
                    if self.debug:
                        log(f"Failed to remove {d}: {e}")
        
        # If debug is not enabled, use a minimal final report.
        final_report = "\n".join(overall_report) if self.debug else "Analysis complete."
        html_report = self.generate_html_report(original_smiles_list, all_tautomer_smiles,
                                                  rank_rdkit_list, rank_xtb_list,
                                                  probabilities_list, tautomer_scores_list,
                                                  final_report)
        
        table = create_output_table(
            original_smiles_list, all_tautomer_smiles,
            rdkit_scores, rank_rdkit_list, rank_xtb_list,
            rel_energy_list, probabilities_list,
            tautomer_counts, threshold_counts, tautomer_scores_list,
            self.deltaG_threshold
        )

        self.finished.emit({
            "results": {"per_molecule": []},
            "report": final_report,
            "table": table,
            "html_report": html_report
        })

    def process_molecule(self, mol_idx, smiles, base_dir, overall_report,
                         original_smiles_list, all_tautomer_smiles,
                         rdkit_scores, rank_rdkit_list, rank_xtb_list,
                         rel_energy_list, probabilities_list,
                         tautomer_counts, threshold_counts,
                         tautomer_scores_list):
        if self.debug:
            overall_report.append(f"\n=== Processing molecule {mol_idx+1}: {smiles} ===")
        original_smiles_list.append(smiles)
        
        try:
            tautomers = enumerate_tautomers(smiles)
        except Exception as e:
            if self.debug:
                overall_report.append(f"Tautomer enumeration error: {e}")
            tautomers = [smiles]
        
        tautomers = tautomers[:self.max_tautomers]
        all_tautomer_smiles.append(tautomers)
        rdkit_score = len(tautomers)
        rdkit_scores.append(rdkit_score)
        if self.debug:
            overall_report.append(f"Found {rdkit_score} tautomers")

        uff_energies = []
        xtb_gibbs_energies = []
        tautomer_scores = []

        for i, taut_smiles in enumerate(tautomers):
            calc_dir = os.path.join(base_dir, f"taut_{i}")
            os.makedirs(calc_dir, exist_ok=True)
            
            try:
                mol3d = generate_3d_structure(taut_smiles, self.method, calc_dir=calc_dir)
                try:
                    uff_energy = calculate_uff_energy(mol3d)
                except Exception:
                    uff_energy = float('inf')
                uff_energies.append(uff_energy)
                taut_score = rdMolStandardize.TautomerEnumerator.ScoreTautomer(mol3d)
                tautomer_scores.append(taut_score)

                xyz_file = os.path.join(calc_dir, "input.xyz")
                write_xyz_file(mol3d, xyz_file)
                log_file = run_xtb(xyz_file, calc_dir, self.optimize, self.solvent, self.thermo)
                energy_Ha = extract_total_energy(log_file)
                xtb_energy = energy_Ha * 627.5095
                xtb_gibbs_energies.append(xtb_energy)
                
                if self.debug:
                    overall_report.append(f"Tautomer {i+1}: Energy {xtb_energy:.2f} kcal/mol")
            except Exception as e:
                overall_report.append(f"Error processing tautomer {i+1}: {str(e)}")
                xtb_gibbs_energies.append(float('inf'))
                uff_energies.append(float('inf'))
                tautomer_scores.append(0)

        n_tauts = len(tautomers)
        if n_tauts > 0 and all(e != float('inf') for e in xtb_gibbs_energies):
            min_energy = min(xtb_gibbs_energies)
            rel_energies = [e - min_energy for e in xtb_gibbs_energies]
            R = 0.0019872041 
            factors = np.exp(-(np.array(rel_energies) / (R * 298.15)))
            probs = factors / np.sum(factors)
        else:
            rel_energies = [0] * n_tauts
            probs = [0] * n_tauts

        rank_rdkit = sorted(range(n_tauts), key=lambda i: uff_energies[i])
        rank_xtb = sorted(range(n_tauts), key=lambda i: xtb_gibbs_energies[i])
        filtered = [i for i, dg in enumerate(rel_energies)]  # Always include all tautomers here

        tautomer_counts.append(n_tauts)
        threshold_counts.append(len(filtered))
        rel_energy_list.append(rel_energies)
        probabilities_list.append(probs)
        rank_rdkit_list.append(rank_rdkit)
        rank_xtb_list.append(rank_xtb)
        tautomer_scores_list.append(tautomer_scores)

    def generate_html_report(self, original_smiles_list, all_tautomer_smiles,
                             rank_rdkit_list, rank_xtb_list,
                             probabilities_list, tautomer_scores_list,
                             final_report):
        try:
            html = f"""
            <html><head><style>
                body {{ font-family: Arial; margin: 20px; }}
                .header {{ color: white; background: #4a90e2; padding: 10px; text-align: center; }}
                .molecule {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                td, th {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                img {{ max-width: 200px; margin: 10px; }}
            </style></head>
            <body>
                <div class="header"><h1>Tautomer Analysis Report</h1></div>
                <pre>{final_report}</pre>
            """
            for mol_idx in range(len(original_smiles_list)):
                html += f"<div class='molecule'><h2>Molecule {mol_idx+1}: {original_smiles_list[mol_idx]}</h2>"
                html += "<table><tr><th>Tautomer</th><th>Properties</th></tr>"
                
                sorted_indices = rank_xtb_list[mol_idx]
                for idx in sorted_indices:
                    taut_smiles = all_tautomer_smiles[mol_idx][idx]
                    mol = Chem.MolFromSmiles(taut_smiles)
                    img = Draw.MolToImage(mol, size=(200,200))
                    buff = io.BytesIO()
                    img.save(buff, format="PNG")
                    img_str = base64.b64encode(buff.getvalue()).decode()
                    
                    html += f"""
                    <tr>
                        <td><img src="data:image/png;base64,{img_str}"></td>
                        <td>
                            SMILES: {taut_smiles}<br>
                            RDKit Score: {tautomer_scores_list[mol_idx][idx]}<br>
                            Probability: {probabilities_list[mol_idx][idx]*100:.2f}%
                        </td>
                    </tr>
                    """
                html += "</table></div>"
            return html + "</body></html>"
        except Exception as e:
            return f"<p>Error generating report: {str(e)}</p>"

# ------------------------------------------------------------
# Main widget with cleanup and debug selection (ΔG slider removed)
# ------------------------------------------------------------
class OWTautomerEnumeration(OWWidget):
    name = "Tautomer EnumerationX"
    description = "Advanced tautomer analysis with RDKit, OpenBabel and xTB"
    icon = "icons/tautomers.png"
    want_main_area = True

    class Inputs:
        data = Input("Data", object, auto_summary=False)

    class Outputs:
        results = Output("Results", dict, auto_summary=False)
        table = Output("Tautomer Table", Table)

    def __init__(self):
        super().__init__()
        self.smiles_list = []
        self.worker = None
        self.last_html = ""

        # GUI setup for 3D Generation method selection
        box_3d = gui.widgetBox(self.controlArea, "3D Generation")
        self.method_radios = [
            QRadioButton("RDKit Conversion"),
            QRadioButton("UFF Minimization"),
            QRadioButton("OpenBabel Conformer Generation")
        ]
        self.method_radios[0].setChecked(True)
        for btn in self.method_radios:
            box_3d.layout().addWidget(btn)

        # xTB Settings
        box_xtb = gui.widgetBox(self.controlArea, "xTB Settings")
        self.optimize_check = QCheckBox("Geometry Optimization")
        self.solvent_combo = QComboBox()
        self.solvent_combo.addItems(["None", "Water", "Octanol", "Hexane"])
        self.thermo_check = QCheckBox("Thermodynamic Corrections")
        box_xtb.layout().addWidget(self.optimize_check)
        box_xtb.layout().addWidget(QLabel("Solvent Model:"))
        box_xtb.layout().addWidget(self.solvent_combo)
        box_xtb.layout().addWidget(self.thermo_check)
        
        # Option to remove temporary directories
        self.remove_temp_check = QCheckBox("Remove temporary directories after computation")
        box_xtb.layout().addWidget(self.remove_temp_check)
        # Option to include debug information
        self.debug_check = QCheckBox("Write debug info")
        box_xtb.layout().addWidget(self.debug_check)

        # Note: The ΔG threshold slider has been removed. A default value is used.
        box_xtb.layout().addWidget(QLabel("Max Tautomers per Molecule:"))
        self.max_taut_spin = QSpinBox()
        self.max_taut_spin.setRange(1, 100)
        self.max_taut_spin.setValue(10)
        box_xtb.layout().addWidget(self.max_taut_spin)

        self.run_btn = gui.button(self.controlArea, self, "Start Analysis",
                                  callback=self.run_enumeration)
        self.pdf_btn = gui.button(self.controlArea, self, "Save PDF Report",
                                  callback=self.generate_pdf)
        self.pdf_btn.setEnabled(False)

        self.report_area = QTextEdit()
        self.report_area.setReadOnly(True)
        self.mainArea.layout().addWidget(self.report_area)

    @Inputs.data
    def set_data(self, data):
        self.smiles_list = []
        if isinstance(data, Table):
            self.smiles_list = get_smiles_list_from_table(data)
            status = f"Loaded {len(self.smiles_list)} molecules from table"
        elif isinstance(data, str):
            self.smiles_list = [data]
            status = f"Loaded single molecule: {data}"
        else:
            status = "Unsupported input type"
        self.report_area.setHtml(f"<b>Input Status:</b> {status}")

    def run_enumeration(self):
        method_map = {
            0: "rdkit_conversion",
            1: "uff_minimisation",
            2: "openbabel_conformer_generation"
        }
        method = method_map[[b.isChecked() for b in self.method_radios].index(True)]
        solvent = self.solvent_combo.currentText().lower()
        solvent = None if solvent == "none" else solvent
        
        self.worker = MultiTautomerWorker(
            smiles_list=self.smiles_list,
            method=method,
            optimize=self.optimize_check.isChecked(),
            solvent=solvent,
            thermo=self.thermo_check.isChecked(),
            max_tautomers=self.max_taut_spin.value(),
            remove_temp=self.remove_temp_check.isChecked(),
            debug=self.debug_check.isChecked()
        )
        
        self.run_btn.setEnabled(False)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_completion)
        self.worker.start()

    def update_progress(self, text):
        self.report_area.setPlainText(text)

    def on_completion(self, results):
        self.run_btn.setEnabled(True)
        self.pdf_btn.setEnabled(True)
        self.last_html = results["html_report"]
        self.report_area.setHtml(self.last_html)
        self.Outputs.results.send(results["results"])
        self.Outputs.table.send(results["table"])

    def generate_pdf(self):
        if not self.last_html:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "", "PDF Files (*.pdf)")
        if filename:
            with open(filename, "wb") as f:
                status = pisa.CreatePDF(self.last_html, dest=f)
            if status.err:
                self.report_area.append("\nPDF generation failed!")
            else:
                self.report_area.append(f"\nSaved PDF report to: {filename}")

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWTautomerEnumeration).run()

