from tensorflow.lite.python.util import convert_bytes_to_c_source
import tensorflow as tf
import tflite_runtime.interpreter as tflite

tflite_model= tf.lite.Interpreter(model_path=args.model_file)
source_text, header_text = convert_bytes_to_c_source(tflite_model,  "sine_model")

with  open('sine_model.h',  'w')  as  file:
    file.write(header_text)

with  open('sine_model.cc',  'w')  as  file:
    file.write(source_text)