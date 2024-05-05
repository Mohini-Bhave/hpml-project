import os
from pathlib import Path
import wandb
from datasets import load_dataset

# Disable wandb for this session
wandb.init(mode="disabled")

# Custom module imports
from model.quantization import ModelQuantizer
from model.student_train import train_student
from model.teacher_train import train_teacher
from evaluation.benchmark import ModelEvaluator
from model.distillation import perform_model_distillation
from model.onnx_model import convert_to_onnx
from plots.plot import plot_model_data
from optimum.onnxruntime import ORTModelForFeatureExtraction

# Setting up environment variable to avoid parallelism issues with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the AG News dataset
ag_news_dataset = load_dataset("ag_news")

# Prepare datasets for training and evaluation
training_data = ag_news_dataset["train"].train_test_split(seed=42)
sampled_train_data = training_data["train"]
student_training_subset = training_data["test"].select(range(1000))
test_data = ag_news_dataset["test"]

# Train models
student_model = train_student("sentence-transformers/paraphrase-MiniLM-L3-v2", sampled_train_data)
teacher_model = train_teacher("sentence-transformers/paraphrase-mpnet-base-v2", sampled_train_data)

# Evaluate both models
evaluator = ModelEvaluator(student_model, test_data)
student_benchmark = evaluator.conduct_benchmark()
teacher_benchmark = evaluator.conduct_benchmark(teacher_model, test_data)

# Perform and evaluate model distillation
distilled_student_model = perform_model_distillation(student_model, teacher_model, student_training_subset)
distilled_student_model.save_pretrained("distilled")

# ONNX conversion for the distilled model
model_directory = Path("distilled")
converted_model, converted_tokenizer = convert_to_onnx(model_directory)

# Benchmarking non-quantized ONNX model
onnx_model_evaluator = ModelEvaluator(converted_model, test_data)
non_quantized_benchmark = onnx_model_evaluator.conduct_benchmark_onnx("onnx/model.onnx")

# Quantize the ONNX model
onnx_quantizer = ModelQuantizer("onnx", student_model.model_head, converted_tokenizer, test_data)
quantized_onnx_model = onnx_quantizer.quantize_model()

# Load and benchmark the quantized ONNX model
quantized_model_evaluator = ModelEvaluator(quantized_onnx_model, test_data)
quantized_model_benchmark = quantized_model_evaluator.conduct_benchmark_onnx("onnx/model_quantized.onnx")

# Final results output
results = {
    "student_model": student_benchmark,
    "teacher_model": teacher_benchmark,
    "distilled_model": distilled_student_model,
    "non_quantized_onnx": non_quantized_benchmark,
    "quantized_onnx": quantized_model_benchmark
}

# Optionally, print the results or use them in further analysis
print(results)