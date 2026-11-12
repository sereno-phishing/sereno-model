"""Training of the Stacking ensemble: Random Forest, XGBoost and LightGBM under a logistic meta-model.

``sereno_model.models.stacking`` holds the method and the saved model class;
``sereno_model.models.train`` reads the training matrix, writes the artifact and its report.
Neither reads the blind test matrix.
"""
