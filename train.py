import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import sys
import argparse

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.utils import set_random_seed
from src.dataset  import train_and_valDataset
from src.model import CrossenhancedCEAM
from src.loss import CenterMSELoss


# training set and validation set
train_id_list = ['003','031','028', '020','040','050','030','042','023','046','013','064','026','056','068','001','055','045','072','052',
                '077','021','047','044','049','029','070','069','038','022','048','067','057','027','043','039','073','066']

val_id_list = ['007', '009', '076', '063', '011', '058', '014', '062', '074', '075']


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
    logging.basicConfig(filename=f'./output_model/{args.save_dir}/debug.log', level=logging.DEBUG)

    train_dataset = train_and_valDataset(args.data_path, train_id_list)
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16,prefetch_factor=2, pin_memory=True)


    val_dataset = train_and_valDataset(args.data_path, val_id_list)
    valloader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=16,prefetch_factor=2, pin_memory=True)
 
    ff_dim = args.embed_dim*4  # Hidden layer size in feed forward network inside transformer
    length = args.core_length + args.extended_length*2 
    max_position_embeddings = length


    model = CrossenhancedCEAM(args.embed_dim, args.num_heads, ff_dim, args.N,args.M,args.K, args.dropout_rate,args.position_embedding_type,max_position_embeddings,args.alpha)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(model)


    train_loss = []
    val_loss = []

    val_loss_flag = float('inf')
    keep_train = 0
    criterion = CenterMSELoss(beta = args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    for epoch in tqdm(range(args.epochs)):
        logging.debug('Epoch [{}/{}], Learning Rate: {:.6f}'.format(epoch+1, args.epochs, optimizer.param_groups[0]['lr']))
        print('Epoch [{}/{}], Learning Rate: {:.6f}'.format(epoch+1, args.epochs, optimizer.param_groups[0]['lr']))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, partnet_inputs, labels = data
            inputs, partnet_inputs, labels = inputs.float().to(device), partnet_inputs.float().to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs,partnet_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss / len(trainloader))

        with torch.no_grad():
            running_loss = 0.0
            for i, data in enumerate(valloader, 0):
                inputs, partnet_inputs,labels = data
                inputs, partnet_inputs,labels = inputs.float().to(device), partnet_inputs.float().to(device), labels.float().to(device)
                outputs = model(inputs,partnet_inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            val_loss.append(running_loss / len(valloader))
            scheduler.step(val_loss[-1])

        logging.debug(f"Epoch {epoch+1}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")
        print(f"Epoch {epoch+1}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")
        if val_loss[-1]<val_loss_flag:
            keep_train = 0
            val_loss_flag = val_loss[-1]
            torch.save(model.state_dict(), f'./output_model/{args.save_dir}/' + args.modality + '.pt')
        else:
            keep_train+=1
        
        if keep_train > 60:
            break



