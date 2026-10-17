# Jetson deployment notes

## Building an engine
Export the model to ONNX first, then build an FP16 engine:
trtexec --onnx=model.onnx --fp16 --saveEngine=model.plan

## Things that bit me
Dynamic shapes need explicit min/opt/max profiles or the engine falls back to the max shape.
Always set the power mode before benchmarking, otherwise latency numbers are not comparable.
Measure with warm-up iterations; the first few runs include lazy initialisation.
