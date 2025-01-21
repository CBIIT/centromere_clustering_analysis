# Centromere Clustering Analysis

This repository accompanies the manuscript **"Simulation and Quantitative Analysis of Spatial Centromere Distribution Patterns"** by Adib Keikhosravi, Krishnendu Guin, Gianluca Pegoraro, and Tom Misteli. It provides tools and scripts for analyzing centromere clustering patterns using high-throughput imaging data and spatial distribution modeling.

## Overview

Centromeres exhibit non-random spatial distribution in the nucleus. Understanding their clustering is critical for studying chromosome behavior, nuclear organization, and associated functional processes. This repository includes:

- **Clustering Metrics**: Tools for analyzing centromere clustering using Ripley's K score, Moran's I, modularity, mean nearest neighbor distance (MNND), etc.
- **Synthetic Data Generation**: Scripts for generating simulated centromere distribution patterns.
- **Modeling Approaches**: Radial and Gaussian-based models for simulating centromere spatial organization.
- **Visualization Tools**: Methods for generating 2D and 3D visualizations of centromere distributions.

## Installation

To install the required dependencies, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/CBIIT/centromere_clustering_analysis.git
   cd centromere_clustering_analysis
2. Install dependencies using the provided `genome.yml` file:
   ```bash
   conda env create -f genome.yml
   conda activate genome
