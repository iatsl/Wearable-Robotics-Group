
1. Create virtualenv
```console
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

2. Dataset preparation

```console
python3 sample_preparation.py --data_folder=<path to the folder with video frames> --save_folder=<path to saave video sequence samples>
```

```console
python3 tfrecords_creation.py --seq_samples_folder=<path to folder with video sequence samples> --save_folder=<path to save tfrecords>
```
