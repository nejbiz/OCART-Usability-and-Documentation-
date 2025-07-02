# OCART-Usability-and-Documentation
Chem-Inf-Widgets for Orange3: A set of custom widgets for Orange3 tailored for chemoinformatics, enabling seamless molecular structure visualization, property calculations, and data analysis. Simplify workflows for cheminformatics and drug discovery with interactive tools.
# OCART a Chemoinformatics extension for Orange
## Setup
### Prerequisites
Before proceeding with the installation, ensure you have the following:
- Python installed on your system(tested on Python 3.12, does not work on Python 3.13) 
- pip(Python package installer) installed

### Instalation steps
1. Download the Software file CHEM_INF_LIGHT.zip 
2. Extract the Software into your desired project folder
3. Open Terminal or Command Prompt
4. Navigate to Project Directory
5. Create a virtual environment `python -m venv venv`
6. Activate virtual environment `source venv\bin\activate`
7. Instal the software: `pip install -e.`
8. Run the software: `python -m Orange.canvas`

## Widgets
### SDF Reader
Analyses an SDF file, it allows selection of properties and reads data based on the selection. There are no inputs. Outputs are selected data tables; list of molecules that have been read and are ready for further processing, analysis, or visualization. 
### SDF Writer
It creates an SDF file from a table with SMILES (Simplified Molecular Input Line Entry System) structures. Inputs is data table and none outputs. 

### ChEMBL Bioactivity Retriever
Fetches bioactivity data with drug design properties. No inputs. Outputs are bioactivity data. 

### Molecular Standardizer
Standardization of molecular structures. It refers to the process of converting chemical structures into consistent molecular representations.  Molecules can be displayed in multiple ways.  

Inputs are original molecules, which are processed through a customized list of standardization actions within the widget. Each output is a standardized form that matches the format of molecules already stored in the database. 

### Molecular Viewer
Displays molecules with optional substructure highlighting and property selection. Inputs are filtered compounds. No outputs (only visualisation tool)

### MolSketcher

Is a USME bsed molecular editor for database creation. No inputs. Outputs: compounds (newly created molecular structures) 

### Substructure Search
It searches compounds based on substructure, superstructure, similarity or exact match. Inputs are input data (a set of molecules to search within).Outputs are filtered compounds (only those that match the search criteria)

### Drug Filter

It filters molecules using Lipinski, PAINS, Veber, QED scores and Reactivity. Inputs are usually input table (input data table containing molecules) -. Outputs are filtered compounds, which meet the selected drug – like criteria 

### MACCS Key Generator
Converts SMILES codes to MACCS keys which are binary molecular fingerprints used for similarity searching and cheminformatics analysis. It also saves them to a table. For inputs are used SMILES Data and for outputs MACCS keys table 
### Fingerprint Calculator_1
Computes molecular fingerprints using RDKit and provides visualizations. Inputs are molecular data and fingerprints are outputs.

### Molecular Similartiy Calculator
Calculates a molecular similarity matrix from fingerprint data using various similarity metrics. 
Inputs are fingerprints (=binary or numerical vectors that represent molecular features) 
Outputs are similarity matrix which is a table showing how similar each molecule is to every other 

### Mordred Descriptors
Compute selected Mordred descriptors for input SMILES. It allows group – based flitering of descriptors. Inputs are SMILES table 
Outputs are descriptors table (numeric values representing chemical properties) 
### QSAR Regression
Users build QSAR (Quantitative Structure–Activity Relationship) 

regression models with flexible settings. It splits the data into training and test sets, supports an optional external set, advanced hyperparameter tuning and provides diagnostic plots. 

Inputs are (external) data and outputs are model, train results, test results and external results 
### Tautomer EnumerationX
It’s advanced tautomer analysis with RDKit, OpenBabel and xTB 
Input: data (molecular) 
Outputs: results and tautomer table; lists all the tautomer variants for each input molecule 
## Example
### 2D Chemical space characterization based on Mordred descriptors for dermal and systemic antimycotics

In the interactive user interface you set up the pipeline of following widgets: SDF Reader -> Mordred descriptors -> PCA -> Scatter plot.
For demonstration puposes calculate the following descriptors: 
- HBAcc: hydrogen-bond acceptor sites​
- NHBDon: hydrogen-bond donor sites​
- nRot: count of rotatable single bonds​
- RotRatio: rotatable bonds ​
- SLogP: predicted log  hydrophobicity​
- SMR — predicted molar refractivity (polarizability)​
- MW — exact MW of the molecule​
- AMW — average exact MW per atom per atom​

In the scatter plot visualize the 2D chemical space for PC1 and PC2. Note that each dataset should be used on each own.
​
