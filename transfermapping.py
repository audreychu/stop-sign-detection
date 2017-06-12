import os, sys
import glob
import pickle

import tensorflow as tf

def getscore(imname):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # change this as you see fit
    image_path = imname

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
            if node_id == 0:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                #print('%s (score = %.5f)' % (human_string, score))
    return score

Xlist = []
for p in sorted(glob.glob('*.jpg'), key=os.path.getmtime):
    if p != 'Picture.jpg':
        Xlist.append(p)

X = []
for path in Xlist:
    with tf.Graph().as_default():
        prob = getscore(path)
        X.append(prob)
    print(path)

with open("heatmaparray.pickle",'wb') as mat:
    pickle.dump(X,mat)
