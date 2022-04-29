import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

CLASSES_MAPPING = {'IS': 0, 'ISLG': 1, 'LG': 2, 'LGIS': 3}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_folder',
        type=str,
        default='tf_records',
        help='path to save tfrecords')
    parser.add_argument(
        '--seq_samples_folder',
        type=str,
        help='path to the folder with frame sequence samples')
    parser.add_argument('--record_name', type=str, help='tf record file name')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_height', type=int, default=256)
    parser.add_argument('--image_width', type=int, default=256)
    parser.add_argument('--image_depth', type=int, default=3)
    return parser.parse_args()


def decode_image(frame_path):
    ''' return image in bytes format and corresponding label '''
    image_bytes = np.asarray(Image.open(frame_path)).tobytes()
    image_bytes = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[image_bytes]))
    label = frame_path.split('/')[-1].split(' ')[-1].split('.jpg')[0]
    label = tf.train.Feature(
        int64_list=tf.train.Int64List(
            value=[
                CLASSES_MAPPING[label]]))
    return image_bytes, label


def get_sequences(video_folder):
    # reading frame pathes
    samples = list()
    for video in list(filter(('.DS_Store').__ne__, os.listdir(video_folder))):
        for sample in os.listdir(os.path.join(video_folder, video)):
            f = open(os.path.join(video_folder, video, sample), 'r')
            data = f.read().splitlines()
            f.close()
            samples.append(data)

    # reading frames and labels
    frame_sequences, label_sequences = list(), list()
    for sample in samples:
        curr_seq, curr_labels = list(), list()
        for frame_path in sample:
            frame, label = decode_image(frame_path)
            curr_seq.append(frame)
            curr_labels.append(label)
        frame_sequences.append(np.array(curr_seq))
        label_sequences.append(np.array(curr_labels))

    return frame_sequences, label_sequences

# TFRecord helpers


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tfrecord_sample(video_data, label, image_size):
    """
        returns anassociated TFRecords example containing the encoded data
    """
    t, h, w, c = len(video_data), image_size[0], image_size[1], image_size[2]

    # save video as list of encoded frames using tensorflow's operation
    img_bytes = [tf.image.encode_jpeg(frame, format='rgb')
                 for frame in video_data]
    with tf.Session() as sess:
        img_bytes = sess.run(img_bytes)

    sequence_dict = {}
    # create a feature for each encoded frame
    img_feats = [
        tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[imgb])) for imgb in img_bytes]
    # save video frames as a FeatureList
    sequence_dict['video_frames'] = tf.train.FeatureList(feature=img_feats)

    # also store associated metadata
    context_dict = {}
    context_dict['label'] = _int64_feature(label)
    context_dict['temporal'] = _int64_feature(t)
    context_dict['height'] = _int64_feature(h)
    context_dict['width'] = _int64_feature(w)
    context_dict['depth'] = _int64_feature(c)

    # combine list + context to create TFRecords example
    sequence_context = tf.train.Features(feature=context_dict)
    sequence_list = tf.train.FeatureLists(feature_list=sequence_dict)
    example = tf.train.SequenceExample(
        context=sequence_context,
        feature_lists=sequence_list)

    return example


def create_tfrecords(
        sample_seqs,
        label_seqs,
        image_size,
        save_folder,
        record_name):
    """Creates a TFRecords dataset from video files.
    Args:
        datafile_path (str): a path to the formatted datafiles (includes train.txt, etc.)
        save_path (str): where to save the .tfrecord files
    """
    os.makedirs(save_folder, exist_ok=True)

    with tf.io.TFRecordWriter(f'{save_folder}/{record_name}.tfrecord') as writer:
        for idx in range(len(sample_seqs)):
            example = tfrecord_sample(
                sample_seqs[idx], label_seqs[idx][-1], image_size)
            writer.write(example.SerializeToString())


def main():
    args = parse_args()
    frame_seqs, label_seqs = get_sequences(args.seq_samples_folder)
    create_tfrecords(frame_seqs,
                     label_seqs,
                     [args.image_height,
                      args.image_width,
                      args.image_depth],
                     args.save_folder,
                     args.record_name)


if __name__ == '__main__':
    main()
