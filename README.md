
# Project Title

Smallholder Tree Cover Classification and Validation Pipeline

# Project Description
This is a modular geospatial and machine learning pipeline designed to generate accurate tree vs. non-tree classification masks for smallholder agricultural plots.

Conventional proprietary models often misclassify vegetation — especially in plots with dense undergrowth, grass, or shrubs — leading to confusion between true tree cover and other vegetation. This project addresses that gap by integrating high-resolution satellite data(Planet), Synthetic aperture rada preprocessing routines, and LiDAR-based validation to produce a robust and interpretable classification model.

## Project Structure
we_love_trees/  
│  
├── data/  
│   ├── geometry/           # AOIs and shapefiles  
│   ├── raw/                # Raw data pulled from Planetary Computer  
│   ├── preprocessed/       # Preprocessed datasets (coregistered, models, etc.)  
│   └── validation/         # Lidar CHM dataset for validation  
│  
├── src/  
│   ├── data/               # Data generation and preprocessing scripts  
│   ├── modelling/          # Model training and classification scripts  
│   ├── utils/              # IO and helper utilities  
│   └── validate/           # Validation scripts (e.g., CHM comparison)  
│  
├── config.py               # Configuration file  
├── pipeline.py             # Orchestrates the preprocessing and modelling workflow  
├── main.py                 # Entry point for running the full pipeline  
├── .gitignore  
├── requirements.txt  
├── README.md   
└── LICENSE  

### Core Objectives

Generate reliable binary masks distinguishing trees from non-trees.

Reduce misclassification caused by undergrowth and mixed vegetation.

Integrate LiDAR-based validation for improved model accuracy assessment.

Build a reusable, modular, and reproducible pipeline for similar land cover studies.




## Installation

Install my-project with pip

First clone this repository
```bash
git clone https://github.com/acreinsightsltd-terp/we_love_trees.git
```
Create a virtual environment
```bash
python -m venv .env
```
Activate the environment
```bash
venv\Scripts\activate (Windows)
source venv/bin/activate(Mac/Linux)
```
Install dependencies
```bash
  pip install -r requirements.txt
```
Run the project
```bash
  python main.py 
```
(Optional) Reproduce specific stages  
Each major stage can also be run independently:
```bash
  python main.py --stage preprocess
  python main.py --stage indices
  python main.py --stage model
  python main.py --stage validate 
```


    
## Authors

- [@KiprutoYG](https://www.github.com/kiprutoYG)

