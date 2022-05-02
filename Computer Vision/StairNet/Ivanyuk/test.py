import os
import sys
from tempfile import template
import tensorflow as tf


def parse_tfr_element(element):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
    sequence_features = {
      'video_frames': tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    }

    context_features = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
      'depth': tf.io.FixedLenFeature([], tf.int64),
      'temporal': tf.io.FixedLenFeature([], tf.int64),
      'label': tf.io.FixedLenFeature([], tf.int64),
    }

    
    content, sequence = tf.io.parse_single_sequence_example(
      element,
      context_features=context_features, 
      sequence_features=sequence_features
    )
  
    height = content['height']
    width = content['width']
    depth = content['depth']
    temporal = content['temporal']
    label = content['label']
    

    #video_frames = list()
    video_data = tf.image.decode_jpeg(tf.gather(sequence['video_frames'], [0])[0])
    #video_data = tf.expand_dims(video_data, 0)
    i = tf.constant(1, dtype=tf.int32)
    # condition of when to stop / loop through every frame
    cond = lambda i, _: tf.less(i, tf.cast(temporal, tf.int32))

    #for i in range(tf.cast(temporal, tf.int32)):
    #  video_frames.append(tf.io.decode_jpeg(tf.gather(sequence['video_frames'], [i])))
    #print(video_frames)
    # reading + decoding the i-th image frame 
    def body(i, video_data):
        # get the i-th index 
        encoded_img = tf.gather(sequence['video_frames'], [i])
        # decode the image 
        img_data = tf.io.decode_jpeg(encoded_img[0]) 
        # append to list using tf operations 
        print(tf.shape(sequence['video_frames']))
        print(tf.shape(video_data))
        print(tf.shape(img_data))
        video_data = tf.concat([video_data, [img_data]], axis=0)
        # update counter & new video_data 
        return (tf.add(i, 1), video_data)

    _, video_data = tf.while_loop(cond, body, [i, video_data], 
            shape_invariants=[i.get_shape(), tf.TensorShape([None])])
    # use this to set the shape + dtype
    video_data = tf.reshape(video_data, shape=[temporal,height,width,depth])
    video_data = tf.cast(video_data, tf.float32)
    #get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(video_data, tf.string)
    feature = tf.reshape(feature, shape=[temporal,height,width,depth])
    return (video_data, label)

def reader(file_path):
    dataset = tf.data.TFRecordDataset(file_path)
    for record in dataset:
        record = tf.train.SequenceExample.FromString(record.numpy())
        yield record

def main():
    # create the dataset 
    file_path = 'data/tfrecords/images.tfrecords'
    print('File exists: ', os.path.exists(file_path))
    #dataset = tf.data.TFRecordDataset(file_path).map(parse_tfr_element)
        #.batch(2)
      
    dataiterator = reader(file_path)

    print()
    print()

    for sample in dataiterator:
      print(tf.io.decode_jpeg(sample))
      print('      ' * 100)

    print('Finished!')

if __name__ == '__main__':
    main()