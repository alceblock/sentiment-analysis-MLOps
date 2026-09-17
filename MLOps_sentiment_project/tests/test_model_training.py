# # Test dataset valid inputs
# def test_dataset_loading():
#     from datasets import load_dataset
#     from model_app.model_utility import DATASET

#     dataset = load_dataset(DATASET, "sentiment", split="train[:5]")

#     assert len(dataset) > 0
#     assert "text" in dataset.column_names
#     assert "label" in dataset.column_names


# # Test compute_metrics function
# def test_compute_metrics():
#     import numpy as np
#     from model_app.model_training import compute_metrics

#     logits = np.array([[0.1,0.8,0.1],
#                        [0.7,0.2,0.1]])

#     labels = np.array([1,0])

#     metrics = compute_metrics((logits, labels))

#     assert "accuracy" in metrics
#     assert "precision" in metrics
#     assert "recall" in metrics
#     assert "f1" in metrics