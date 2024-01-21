import numpy as np
import gc
import os
import json
import argparse
from src.utils import set_random_seed

def parse_arguments():
    parser = argparse.ArgumentParser(description='data proprecessing Script Description')

    parser.add_argument('--seed_value', type=int, default=42, 
                        help='Random seed')
    args = parser.parse_args()
    return args

def Normalizedata(train_data,val_data,save_name=None):
    X = np.array(train_data + val_data)
    min_ = np.min(X, axis=0)
    max_ = np.max(X, axis=0)
    X = None
    gc.collect()


    has_nan = np.isnan(np.array(train_data) - min_).any()

    print(has_nan)

    has_nan = np.isnan(np.array(val_data) - min_).any()
    print(has_nan)
    

    a = max_ - min_
    b = a.copy()
    b[a == 0] = max_[a == 0]  
    b[b == 0] = 1.0

    has_zero = np.any(b == 0)


    print(has_zero)

    train_data_normalized = (np.array(train_data) - min_) / b
    val_data_normalized = (np.array(val_data) - min_) / b
    print("normalized train/val data end")

    train_data = None
    val_data = None
    gc.collect()

    np.save("./data/train_data_normalized.npy", train_data_normalized)
    np.save("./data/val_data_normalized.npy", val_data_normalized)
    gc.collect()
    
def Normalizetestdata(train_data,val_data,test_data,save_name=None):
    X = np.array(train_data + val_data+test_data)
    min_ = np.min(X, axis=0)
    max_ = np.max(X, axis=0)
    X = None
    gc.collect()

    has_nan = np.isnan(np.array(train_data) - min_).any()
    print(has_nan)

    has_nan = np.isnan(np.array(val_data) - min_).any()
    print(has_nan)
    
    has_nan = np.isnan(np.array(test_data) - min_).any()
    print(has_nan)

    a = max_ - min_
    b = a.copy()
    b[a == 0] = max_[a == 0] 
    b[b == 0] = 1.0 

    has_zero = np.any(b == 0)

    print(has_zero)

    test_data_normalized = (np.array(test_data) - min_) / b
    print("normalized test data end")

    train_data = None
    val_data = None
    gc.collect()

    np.save("./data/test_data_normalized.npy", test_data_normalized)
    gc.collect()

if __name__ == "__main__":
    args = parse_arguments()
    set_random_seed(args.seed_value)

    train_dir = r"../noxi/train"

    modalities = [
        ".audio.gemaps.stream~", 
        ".audio.soundnet.stream~", 
        ".video.openface2.stream~",
        ".kinect2.skeleton.stream~",
        ".video.openpose.stream~",
        ".kinect2.au.stream~"
        ]
    modalities_dim = [
        58, 
        256, 
        673,
        350, 
        350,
        17
        ]


    train_anno = {}

    for path, sub_dirs, files in os.walk(train_dir):
        for f in files:
            session_id = path.split("/")[-1]
            role = f.split(".")[0]
            key = role + ";" + session_id
            if ".engagement.annotation.csv" in f:
                train_anno[key] = os.path.join(os.path.join(path), f)


    train_data = []
    train_labels = []
    train_class_data = {}
    val_class_data = {}
    test_class_data = {}

    for entry in train_anno:
        features = {}
        lengths = []
        mod_name = []
        for idx, modality in enumerate(modalities):
            f = open("/".join(train_anno[entry].split("/")[0:-1:1]) + "/" + entry.split(";")[0] + modality)
            a = np.fromfile(f, dtype=np.float32)
            a = a.reshape((int(a.shape[0] / modalities_dim[idx]), modalities_dim[idx]))
            features[modality] = a
            lengths.append(a.shape[0])
            mod_name.append((modality, modalities_dim[idx]))

        anno_file = np.genfromtxt(train_anno[entry], delimiter="\n", dtype=str)
        lengths.append(len(anno_file))
        num_samples = min(lengths)

        print(entry,lengths)

        train_class_data[entry] = num_samples
        for i in range(num_samples):
            sample = None
            for idx, modality in enumerate(modalities):
                if sample is None:
                    sample = np.nan_to_num(features[modality][i])
                else:
                    sample = np.concatenate([sample, np.nan_to_num(features[modality][i])])

            train_labels.append(float(anno_file[i]))
            train_data.append(sample)


    print("train data loaded")


    val_dir = r"../noxi/val"

    val_anno = {}

    for path, sub_dirs, files in os.walk(val_dir):
        for f in files:
            session_id = path.split("/")[-1]
            role = f.split(".")[0]
            key = role + ";" + session_id
            if ".engagement.annotation.csv" in f:
                val_anno[key] = os.path.join(os.path.join(path), f)

    val_data = []
    val_labels = []

    for entry in val_anno:
        features = {}
        lengths = []
        for idx, modality in enumerate(modalities):
            f = open("/".join(val_anno[entry].split("/")[0:-1:1]) + "/" + entry.split(";")[0] + modality)
            a = np.fromfile(f, dtype=np.float32)
            a = a.reshape((int(a.shape[0] / modalities_dim[idx]), modalities_dim[idx]))
            features[modality] = a
            lengths.append(a.shape[0])

        anno_file = np.genfromtxt(val_anno[entry], delimiter="\n", dtype=str)
        lengths.append(len(anno_file))
        num_samples = min(lengths)

        print(entry,lengths)

        val_class_data[entry] = num_samples
        for i in range(num_samples):
            sample = None
            for idx, modality in enumerate(modalities):
                if sample is None:
                    sample = np.nan_to_num(features[modality][i])
                else:
                    sample = np.concatenate([sample, np.nan_to_num(features[modality][i])])
            val_labels.append(float(anno_file[i]))
            val_data.append(sample)


    print("val data loaded")


    test_dir = r"../noxi/test"

    test_anno = {}

    for path, sub_dirs, files in os.walk(test_dir):
        for f in files:
            session_id = path.split("/")[-1]
            role = f.split(".")[0]
            key = role + ";" + session_id
            # if ".engagement.annotation.csv" in f:
            test_anno[key] = os.path.join(os.path.join(path), f)

    print(test_anno)
    test_data = []
    #test_labels = []


    for entry in test_anno:
        features = {}
        lengths = []
        for idx, modality in enumerate(modalities):
            
            f = open("/".join(test_anno[entry].replace("private", "public").split("/")[0:-1:1]) + "/" + entry.split(";")[0] + modality)
            a = np.fromfile(f, dtype=np.float32)
            a = a.reshape((int(a.shape[0] / modalities_dim[idx]), modalities_dim[idx]))
            features[modality] = a
            lengths.append(a.shape[0])
        
        # anno_file = np.genfromtxt(test_anno[entry], delimiter="\n", dtype=str)
        # lengths.append(len(anno_file))
        num_samples = min(lengths)

        print(entry,lengths)

        test_class_data[entry] = num_samples
        
        for i in range(num_samples):
            sample = None
            for idx, modality in enumerate(modalities):
                if sample is None:
                    sample = np.nan_to_num(features[modality][i])
                else:
                    sample = np.concatenate([sample, np.nan_to_num(features[modality][i])])
            #test_labels.append(float(anno_file[i]))
            test_data.append(sample)
            

    print("test data loaded")

    train_class_data_json_str = json.dumps(train_class_data)


    with open('./data/train_class_dict.json', 'w') as file:
        file.write(train_class_data_json_str)
    val_class_data_json_str = json.dumps(val_class_data)


    with open('./data/val_class_dict.json', 'w') as file:
        file.write(val_class_data_json_str)
    print("json data end")

    test_class_data_json_str = json.dumps(test_class_data)
    with open('./data/test_class_dict.json', 'w') as file:
        file.write(test_class_data_json_str)
    print("json data end")


    np.save("./data/train_labels.npy", np.asarray(train_labels))
    np.save("./data/val_labels.npy", np.asarray(val_labels))
    # np.save("./data/test_labels.npy", np.asarray(test_labels))
    Normalizedata(train_data,val_data)
    Normalizetestdata(train_data,val_data,test_data)


