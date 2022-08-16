from sparsezoo.models import Zoo
from deepsparse import compile_model

batch_size = 16
stub = "zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95"

# Download model and compile as optimized executable for your machine
model = Zoo.download_model_from_stub(stub, override_parent_path="downloads")
engine = compile_model(model, batch_size=batch_size)

# Runs a benchmark
inputs = model.data_inputs.sample_batch(batch_size=batch_size)
benchmarks = engine.benchmark(inputs)
print(benchmarks)
