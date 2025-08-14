# TIE MCP Server Examples

This directory contains example notebooks and scripts demonstrating how to use and extend the TIE MCP Server.

## Available Examples

### model_retraining.ipynb
An interactive Jupyter notebook that demonstrates:
- Loading and exploring TIE datasets
- Creating custom datasets with filtering and augmentation
- Training different model types (WALS, BPR, Top Items)
- Evaluating model performance
- Comparing model configurations
- Preparing models for deployment

#### Prerequisites
To run the notebook, ensure you have Jupyter installed:
```bash
poetry install --with dev
poetry run jupyter notebook examples/model_retraining.ipynb
```

#### Key Features
- **Dataset Analysis**: Visualize technique distributions and co-occurrence patterns
- **Data Preprocessing**: Filter techniques by frequency and apply augmentation
- **Model Training**: Experiment with different algorithms and hyperparameters
- **Performance Evaluation**: Compare models using metrics like NDCG, precision, and recall
- **Deployment Preparation**: Generate configuration files for production deployment

## Running Examples

1. Install the development dependencies:
   ```bash
   poetry install --with dev
   ```

2. Start Jupyter:
   ```bash
   poetry run jupyter notebook
   ```

3. Navigate to the `examples` directory and open the desired notebook.

## Contributing Examples

If you have useful examples or notebooks that demonstrate TIE MCP Server capabilities, please consider contributing them. Ensure that:
- Examples are well-documented with clear explanations
- Any API keys or sensitive data are replaced with placeholders
- Dependencies are clearly listed
- Code follows the project's style guidelines

## Note on Data Collection

For security reasons, data collection scripts with actual API endpoints should not be included in the repository. Instead, provide templates with placeholder values that users can customize with their own credentials.