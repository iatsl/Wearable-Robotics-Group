import os
import argparse

CLASSES = ['IS', 'ISLG', 'LG', 'LGIS']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='dataset folder path')
    parser.add_argument('--save_folder', type=str, default='data/seq_samples',
                        help='sequence samples folder path')
    parser.add_argument(
        '--seq_size',
        type=int,
        default=4,
        help='sample sequence size')
    parser.add_argument(
        '--pad_type',
        choices=[
            'copy',
            'random',
            'zeros',
            'ones'],
        default='copy',
        type=str,
        help="padding type for sequence that don't have enough frames to construct a sequence")
    return parser.parse_args()


def get_video_number(file_name):
    '''
        parse video number from string
        input: [IMG_#_#] frame # #CLASS#.jpg
        output: IMG_#_#
    '''
    return file_name.split(' ')[0].replace("['", '').replace("']", '')


def get_class(filename):
    '''
        parse frame label from filename
    '''
    return filename.split(' ')[-1].split('.jpg')[0]


def get_frame_samples(
        class_folder_names,
        filter_ds_store=True):
    '''
        get list of all video frames
    '''
    samples = list()
    for folder in class_folder_names:
        for el in os.listdir(folder):
            samples.append(el)
    if filter_ds_store:
        return list(filter(('.DS_Store').__ne__, samples))
    return samples


def create_video2frame_dict(video_samples, data_folder):
    ''' mapping video number to corresponding frames '''
    video_names_dict = {get_video_number(el): list() for el in video_samples}

    # creating folder for each video
    for name in video_names_dict.keys():
        dir_name = os.path.join(data_folder, name)
        os.makedirs(dir_name, exist_ok=True)
        print(f'\t Created video folder: {dir_name}')

    # mapping video with corresponding sorted frames
    for image_path in sorted(
            video_samples, key=lambda x: int(
            x.split(' ')[2])):
        video_n = get_video_number(image_path)
        video_names_dict[video_n].append(image_path)

    return video_names_dict


def construct_sequence_samples(
        video_dict,
        data_folder,
        save_folder,
        seq_size,
        pad_type):
    '''
        create sequence samples of length seq_size
    '''
    for folder_name, image_seq in video_dict.items():
        for i in range(0, len(image_seq) - seq_size):
            if i < seq_size:
                if pad_type == 'copy':
                    subset = [image_seq[0] for _ in range(seq_size - i - 1)]
                elif pad_type in ['random', 'zeros', 'ones']:
                    subset = [pad_type for _ in range(seq_size - i - 1)]
                else:
                    raise NotImplementedError
                subset.extend(image_seq[max(0, i - seq_size): i + 1])
            else:
                subset = image_seq[i - seq_size: i]

            with open(os.path.join(save_folder, folder_name, f'sample_{i}.txt'), 'w') as f:
                f.write('\n'.join(
                    [os.path.join(data_folder, get_class(el), el) for el in subset]))


def main():
    args = parse_args()

    class_folder_names = [
        os.path.join(args.data_folder, class_name) for class_name in CLASSES
    ]
    print('Class folders: ', class_folder_names)

    samples = get_frame_samples(class_folder_names)
    print('Number of samples: ', len(samples))

    video_dict = create_video2frame_dict(samples, args.save_folder)
    print('Number of videos: ', len(video_dict))

    construct_sequence_samples(
        video_dict,
        args.data_folder,
        args.save_folder,
        args.seq_size,
        args.pad_type)


if __name__ == '__main__':
    main()
