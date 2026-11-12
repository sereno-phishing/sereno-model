"""Feature vector of a URL: one definition shared by training and by the inference server.

Nothing is imported here on purpose: ``sereno_model.features.extract`` must stay importable
with the standard library and ``tldextract`` only, while ``sereno_model.features.build`` is
training-side and pulls in pandas and pyarrow.
"""
