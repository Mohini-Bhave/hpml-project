from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
import functools
import evaluate
import onnxruntime
from optimum.onnxruntime import ORTModelForFeatureExtraction
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from setfit.exporters.utils import mean_pooling

def onnx_conversion(model_id):
    onnx_path = Path("onnx")
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        model_id, from_transformers=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ort_model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)
    return ort_model, tokenizer









