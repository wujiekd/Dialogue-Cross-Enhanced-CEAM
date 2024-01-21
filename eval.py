import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import json
import pandas as pd
import argparse



from src.utils import set_random_seed
from src.dataset import testDataset
from src.model import CrossenhancedCEAM
from src.metric import concordance_correlation_coefficient

def parse_arguments():
    parser = argparse.ArgumentParser(description='CEAM Script Description')

    parser.add_argument('--save_dir', type=str, default='Cross_CEAM', 
                        help='Save directory name')
    parser.add_argument('--position_embedding_type', type=str, default='fixed', 
                        help='Position embedding type')
    parser.add_argument('--modality', type=str, default='multimodal', 
                        help='Modality')
    parser.add_argument('--data_path', type=str, default='./data_96_32/all_data', 
                        help='Data path')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs')
    parser.add_argument('--N', type=int, default=1, 
                        help='N')
    parser.add_argument('--M', type=int, default=1, 
                        help='M')
    parser.add_argument('--K', type=int, default=2, 
                        help='K')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='weighted block skip connection')
    parser.add_argument('--beta', type=float, default=0.5, 
                        help='Center Mse loss')                    
    parser.add_argument('--learning_rate', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=768, 
                        help='Embedding dimension for attention')
    parser.add_argument('--num_heads', type=int, default=8, 
                        help='Number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.0, 
                        help='Dropout rate')
    parser.add_argument('--core_length', type=int, default=32, 
                        help='Core length')
    parser.add_argument('--extended_length', type=int, default=32, 
                        help='Extended length')
    parser.add_argument('--seed_value', type=int, default=42, 
                        help='Random seed')
    
    args = parser.parse_args()
    return args


 

if __name__ == "__main__":
    args = parse_arguments()
    set_random_seed(args.seed_value)

    
    ff_dim = args.embed_dim*4  # Hidden layer size in feed forward network inside transformer
    length = args.core_length + args.extended_length*2 
    max_position_embeddings = length


    eval_model = CrossenhancedCEAM(args.embed_dim, args.num_heads, ff_dim, args.N,args.M,args.K, args.dropout_rate,args.position_embedding_type,max_position_embeddings,args.alpha)
    eval_model.load_state_dict(torch.load(f'./output_model/{args.save_dir}/' + args.modality + '.pt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_model = eval_model.to(device)
    eval_model.eval()

    val_labels = np.load(f"./data/val_labels.npy") 

    with open(f"./data/val_class_dict.json", 'r') as file:
        val_class_data = json.load(file)

    all_num = 0
    for key, value in val_class_data.items():
        all_num += value


    allpred = np.zeros(all_num)  

    start_idx = 0
    for attr, count in val_class_data.items():
        name = attr.split(';')[0]
        id = attr.split(';')[1]
        val_dataset = testDataset(args.data_path,id,name)
        valloader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=16,prefetch_factor=2, pin_memory=True)
        
        ypred = []
        inputs_ = []
        labels_ = []
        step = args.core_length
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):

                inputs,add_data, labels = data
                inputs= inputs.float().to(device)
                add_data= add_data.float().to(device)
                outputs = eval_model(inputs,add_data)
                ypred.append(outputs.cpu().numpy())

                inputs_.append(inputs)
                labels_.append(labels)
                
            ypred = np.concatenate(ypred, axis=0)
            reshaped_data = ypred.reshape(ypred.shape[0],ypred.shape[1])
            labels_ = np.concatenate(labels_, axis=0)
            
            if (count - step) // step != 0:
                n_samples = (count - step) // step + 1
            else:n_samples = (count - step) // step 

            predict = np.zeros(n_samples*step+step)
            for j in range(reshaped_data.shape[0]):
                start = j * step
                end = start + step
                predict[start:end] += reshaped_data[j][step:step*2]

            allpred[start_idx:start_idx+count] = predict[:count]
            start_idx+=count
            print(start_idx)

    print("multimodal : " + str(concordance_correlation_coefficient(allpred, np.asarray(val_labels))))
    print("eval validation end~~~~~~")



    output_path = f'./output_test/{args.save_dir}' 
    with open(f"./data/test_class_dict.json", 'r') as file:
        test_class_data = json.load(file)

    all_num = 0
    for key, value in test_class_data.items():
        all_num += value

    allpred = np.zeros(all_num)  


    start_idx = 0
    for attr, count in test_class_data.items():
        name = attr.split(';')[0]
        id = attr.split(';')[1]
        val_dataset = testDataset(args.data_path,id,name,False)
        valloader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=16,prefetch_factor=2, pin_memory=True)
        
        ypred = []
        inputs_ = []
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):

                inputs,add_data = data
                inputs= inputs.float().to(device)
                add_data= add_data.float().to(device)
                outputs = eval_model(inputs,add_data)
                ypred.append(outputs.cpu().numpy())

                inputs_.append(inputs)
            
                
            ypred = np.concatenate(ypred, axis=0)
            reshaped_data = ypred.reshape(ypred.shape[0],ypred.shape[1])
        
            if (count - step) // step != 0:
                n_samples = (count - step) // step + 1
            else:n_samples = (count - step) // step 

            predict = np.zeros(n_samples*step+step)
            for j in range(reshaped_data.shape[0]):
                start = j * step
                end = start + step
                predict[start:end] += reshaped_data[j][step:step*2]
    
            allpred[start_idx:start_idx+count] = predict[:count]
            output = predict[:count]
            start_idx+=count
            print(start_idx)

            df = pd.DataFrame(output)
            if not os.path.exists(f'{output_path}/{id}'):
                os.makedirs(f'{output_path}/{id}')

            df.to_csv(f'{output_path}/{id}/{name}.engagement.annotation.csv', header=False, index=False)


     