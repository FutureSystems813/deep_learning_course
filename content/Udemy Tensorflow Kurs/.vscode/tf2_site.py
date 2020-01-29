print("Setting tf.keras as keras module.")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Show: WARNINGS, ERRORS
import tensorflow
sys.modules["keras"] = tensorflow.keras
sys.modules["keras.advanced_activations"] = tensorflow.keras.activations
sys.modules["keras.utils.np_utils"] = tensorflow.keras.utils
sys.modules["tensorflow.keras.utils.np_utils"] = tensorflow.keras.utils