import numpy as np
import os
import json
import argparse
from src.utils import set_random_seed

def parse_arguments():
    parser = argparse.ArgumentParser(description='CSW Script Description')

    parser.add_argument('--core_length', type=int, default=32, 
                        help='Core length')
    parser.add_argument('--extended_length', type=int, default=32, 
                        help='Extended length')
    parser.add_argument('--input_path', type=str, default='./data', 
                        help='Raw data path')
    parser.add_argument('--seed_value', type=int, default=42, 
                        help='Random seed')
    args = parser.parse_args()
    return args



def split_feature(data,save_dir,name):
    if (data.shape[0] ) // step != 0:
        n_samples = data.shape[0]  // step + 1
    else:n_samples = data.shape[0] // step 
    
    for i in range(n_samples):
        start = i * step - step
        end = start + length
        if start < 0:
            zeros_y = np.zeros((step, 1704 ))
            padded_data = np.concatenate((zeros_y,data[:end]))
            save_data = padded_data
        elif end > data.shape[0]:
            pad_length = end - data.shape[0]
            zeros_y = np.zeros((pad_length, 1704 ))
            padded_data = np.concatenate((data[start:], zeros_y))
            save_data = padded_data
        else:
            save_data = data[start:end]
    
        np.save(f"{save_dir}/frame_feature_{name}_{i}.npy", save_data)

def split_label(data,save_dir,name):
    if (data.shape[0] ) // step != 0:
        n_samples = data.shape[0]  // step + 1
    else:n_samples = data.shape[0] // step 
  
    for i in range(n_samples):
        start = i * step - step
        end = start + length
        if start < 0:
            zeros_y = np.zeros((step,))
            padded_data = np.concatenate((zeros_y,data[:end]))
            save_data = padded_data
        elif end > data.shape[0]:
            pad_length = end - data.shape[0]
            zeros_y = np.zeros((pad_length, ))
            padded_data = np.concatenate((data[start:], zeros_y))
            save_data = padded_data
        else:
            save_data = data[start:end]

        np.save(f"{save_dir}/label_{name}_{i}.npy", save_data)
    
def split_per_data(all_data,all_label,class_data):
    start_index = 0
    end_index = 0
    for attr, count in class_data.items():
        print(attr,count)
        name = attr.split(';')[0]
        save_dir = os.path.join(output_path,str(attr.split(';')[1]))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        end_index += count
        split_feature(all_data[start_index:end_index],save_dir,name)
        if all_label is not None:
            split_label(all_label[start_index:end_index],save_dir,name)
        start_index += count


train_id_list = ['003','031','028', '020','040','050','030','042','023','046','013','064','026','056','068','001','055','045','072','052',
                  '077','021','047','044','049','029','070','069','038','022','048','067','057','027','043','039','073','066']

val_id_list = ['007', '009', '076', '063', '011', '058', '014', '062', '074', '075']


if __name__ == "__main__":
    args = parse_arguments()
    set_random_seed(args.seed_value)

    length = args.core_length + args.extended_length * 2
    step = args.core_length

    target_name = f'data_{length}_{step}'

    input_path = args.input_path
    output_path = f'./{target_name}/all_data'


    X_val = np.load(f"{input_path}/val_data_normalized.npy")
    val_labels = np.load(f"{input_path}/val_labels.npy")
    with open(f"{input_path}/val_class_dict.json", 'r') as file:
        val_class_data = json.load(file)
        
    split_per_data(X_val,val_labels,val_class_data)

    X_train = np.load(f"{input_path}/train_data_normalized.npy")
    y_train = np.load(f"{input_path}/train_labels.npy")
    with open(f"{input_path}/train_class_dict.json", 'r') as file:
        train_class_data = json.load(file)

    split_per_data(X_train,y_train,train_class_data)


    X_test = np.load(f"{input_path}/test_data_normalized.npy")
    y_test = None
    with open(f"{input_path}/test_class_dict.json", 'r') as file:
        test_class_data = json.load(file)

    split_per_data(X_test,y_test,test_class_data)


