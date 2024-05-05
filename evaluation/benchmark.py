from tqdm.auto import tqdm
import evaluate
import numpy as np
import torch
from time import perf_counter
from pathlib import Path

# Define a dictionary for categorizing texts
category_codes = {"1": "World", "2": "Sci/Tech", "3": "Politics", "4": "Business"}
# Initialize the accuracy metric using the 'evaluate' library
accuracy_evaluator = evaluate.load("accuracy")

class ModelBenchmark:
    def __init__(self, model, dataset):
        self.dataset = dataset
        self.model = model

    def compute_accuracy(self):
        # Get model predictions and corresponding ground truth labels
        predicted_labels = self.model.predict(self.dataset["text"])
        true_labels = self.dataset["label"]
        # Calculate and return the model's accuracy
        result_accuracy = accuracy_evaluator.compute(predictions=predicted_labels, references=true_labels)['accuracy']
        return result_accuracy

    def calculate_model_size(self):
        # Save the model's state dictionary temporarily to measure its size
        model_state = self.model.model_body.state_dict()
        temp_model_file = Path("temp_model.pt")
        torch.save(model_state, temp_model_file)
        # Determine the file size in megabytes
        size_in_mb = temp_model_file.stat().st_size / (1024 ** 2)
        temp_model_file.unlink()  # Clean up the temporary file
        return size_in_mb

    def evaluate_latency(self, sample_text="Is this a test?"):
        latency_measurements = []
        # Run a few warm-up iterations to stabilize performance
        for _ in range(20):
            self.model([sample_text])
        # Record the time taken for each prediction
        for _ in range(200):
            start_time = perf_counter()
            self.model([sample_text])
            end_time = perf_counter()
            latency_measurements.append(end_time - start_time)
        # Calculate and return average latency in milliseconds
        average_latency = np.mean(latency_measurements) * 1000
        return average_latency

    def run_benchmark(self):
        benchmark_results = {
            "accuracy": self.compute_accuracy(),
            "latency": self.evaluate_latency(),
            "model size": self.calculate_model_size()
        }
        return benchmark_results

    def run_benchmark_onnx(self, onnx_model, model_path):
        self.model = onnx_model
        self.onnx_model_path = model_path
        onnx_results = {
            "accuracy": self.compute_accuracy_onnx(onnx_model),
            "latency": self.evaluate_latency(),
            "model size": self.calculate_size_onnx()
        }
        return onnx_results

    def compute_accuracy_onnx(self, onnx_model):
        predictions = []
        batch_size = 100
        for start_index in tqdm(range(0, len(self.dataset["text"]), batch_size)):
            batch_predictions = onnx_model.predict(self.dataset["text"][start_index:start_index + batch_size])
            predictions.extend(batch_predictions)
        ground_truths = self.dataset["label"]
        accuracy_onnx = accuracy_evaluator.compute(predictions=predictions, references=ground_truths)['accuracy']
        return accuracy_onnx

    def calculate_size_onnx(self):
        model_file_size = Path(self.onnx_model_path).stat().st_size / (1024 ** 2)
        return model_file_size