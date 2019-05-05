import timeit
import math

from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import tensorflow as tf

all_batch_size = 1
np.random.seed(seed=69)

def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)

modelname = 'resources/networks/mars-small128'

# OV configuration
ov_net = IENetwork(model=modelname + '.xml', weights=modelname + '.bin')
ov_net.batch_size = all_batch_size
ov_plugin = IEPlugin(device='CPU')

# TF configuration
tf_session = tf.Session()
with tf.gfile.GFile(modelname + '.pb', 'rb') as gfile:
    tf_graph = tf.GraphDef()
    tf_graph.ParseFromString(gfile.read())
tf.import_graph_def(tf_graph, name='net')
tf_input_node = tf.get_default_graph().get_tensor_by_name('net/images:0')
tf_output_node = tf.get_default_graph().get_tensor_by_name('net/features:0')


# ?x128x64x3
testinput = np.random.random_sample((all_batch_size, 128, 64, 3))
testinput2 = testinput[:, :, :, ::-1]
print(testinput - testinput2)
# openvino expects colors major
ov_testinput = np.transpose(testinput, (0, 3, 1, 2))
ov_testinput2 = np.transpose(testinput2, (0, 3, 1, 2))

# run OV
ov_input_blob = next(iter(ov_net.inputs))
ov_out_blob = next(iter(ov_net.outputs))
ov_exec_net = ov_plugin.load(network=ov_net)

def run_ov(inp):
    return ov_exec_net.infer(inputs={ov_input_blob: inp})

ov_res = next(iter(run_ov(ov_testinput).values()))
ov_res2 = next(iter(run_ov(ov_testinput2).values()))

# run TF
def run_tf(inp):
    tf_output = np.zeros((all_batch_size, 128), np.float32)
    _run_in_batches(lambda x: tf_session.run(tf_output_node, feed_dict=x),
                    {tf_input_node: inp}, tf_output, all_batch_size)
    return tf_output

tf_res = run_tf(testinput)
tf_res2 = run_tf(testinput2)

def compare(vec1, vec2):
    print('Diff abs (0.0 is exactly same):\n', vec1 - vec2)
    print('Diff rel (1.0 is exactly same):\n', vec1 / vec2)

    comp = 'PASSED' if np.allclose(vec1, vec2) else 'FAILED'
    print('Comparison: {}'.format(comp))

# compare different results
print('TF: RGB vs BGR')
compare(tf_res, tf_res2)
print('')

print('OV: RGB vs BGR')
compare(ov_res, ov_res2)
print('')

print('TF vs OV')
compare(tf_res, ov_res)
print('')

# timing
iterations = int(300 / all_batch_size)
print('Batch size {}, {} iterations:'.format(all_batch_size, iterations))
print(' OV: {:.5f}s'.format(timeit.timeit('run_ov(ov_testinput)', number=iterations, globals=globals())))
print(' TF: {:.5f}s'.format(timeit.timeit('run_tf(testinput)', number=iterations, globals=globals())))