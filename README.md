# Oil Production Prediction Model

This model combines __exponential regression__ with __K-nearest neighbors__ to predict the quantity of oil production from wells drilled in any area of northern Colorado. Without access to proprietary geological data, it uses exclusively public data downloaded or scraped from Colorado's [official ECMC website](https://ecmc.state.co.us/).

I've included a [demo notebook](demo/demo.ipynb) to show how it works, including hyperparameter tuning with grid search and 10-fold cross-validation and evaluation.