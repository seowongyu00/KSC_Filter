import json
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict
from lion_pytorch import Lion

def seed_fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

def collate_fn(batch):
    max_length = 5
    input_x_list, output_x_list, task_list = zip(*batch)
    batch_size = len(batch)
    input_x_tensor = torch.full((batch_size, max_length, 30, 30), 0)
    output_x_tensor = torch.full((batch_size, max_length, 30, 30), 0)
    task_tensor = torch.full((batch_size,), -1, dtype=torch.long)

    for i in range(batch_size):
        input_seq = input_x_list[i]
        output_seq = output_x_list[i]
        seq_length = min(max_length, len(input_seq))
        
        if not isinstance(input_seq[0], torch.Tensor):
            input_seq = [torch.tensor(seq) for seq in input_seq]
            output_seq = [torch.tensor(seq) for seq in output_seq]

        input_x_tensor[i, :seq_length] = torch.cat([seq.unsqueeze(0) for seq in input_seq[:seq_length]], dim=0)
        output_x_tensor[i, :seq_length] = torch.cat([seq.unsqueeze(0) for seq in output_seq[:seq_length]], dim=0)
        task_tensor[i] = torch.tensor(task_list[i], dtype=torch.long).clone().detach()
        
    return input_x_tensor, output_x_tensor, task_tensor

class New_ARCDataset(Dataset):
    def __init__(self, file_name, mode='task'):
        self.dataset = None
        self.count_boundary = 2500
        self.count = 0
        self.mode = mode

        with open(file_name, 'r') as f:
            self.dataset = json.load(f)

        if self.mode == 'task':
            task_list = list(set(self.dataset['task']))
            self.task_dict = {task: i for i, task in enumerate(task_list)}

    def __len__(self):
        return len(self.dataset['input'])

    def __getitem__(self, idx):
        x = self.dataset['input'][idx]
        y = self.dataset['output'][idx]

        if self.mode == 'task':
            task = self.task_dict[self.dataset['task'][idx]]
        else:
            task = None

        return torch.tensor(x), torch.tensor(y), torch.tensor(task)
    
class vae_Linear_origin(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(11, 512)

        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(128, 128)
        self.sigma_layer = nn.Linear(128, 128)
        self.proj = nn.Linear(512, 11)

    def forward(self, x):
        if len(x.shape) > 3:
            batch_size = x.shape[0]
            embed_x = self.embedding(x.reshape(batch_size, 5*900).to(torch.long))
        else:
            embed_x = self.embedding(x.reshape(1, 5*900).to(torch.long))
        feature_map = self.encoder(embed_x)
        mu = self.mu_layer(feature_map)
        sigma = self.sigma_layer(feature_map)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        latent_vector = mu + std * eps
        output = self.decoder(latent_vector)
        output = self.proj(output).reshape(-1,30,30,11)

        return output

class new_idea_vae(nn.Module):
    def __init__(self, model_file):
        super().__init__()

        self.autoencoder = vae_Linear_origin()
        self.autoencoder.load_state_dict(torch.load(model_file))
        self.auto_encoder_freeze()

        self.first_layer_parameter_size = 256
        self.second_layer_parameter_size = 128
        self.third_layer_parameter_size = 64
        self.fourth_layer_parameter_size = 32
        self.last_parameter_size = 16
        self.num_categories = 1

        self.fusion_layer1 = nn.Linear(2*128*5*900, self.first_layer_parameter_size)
        self.fusion_layer2 = nn.Linear(self.first_layer_parameter_size, self.second_layer_parameter_size)
        self.fusion_layer3 = nn.Linear(self.second_layer_parameter_size, self.third_layer_parameter_size)
        self.fusion_layer4 = nn.Linear(self.third_layer_parameter_size, self.fourth_layer_parameter_size)
        self.fusion_layer5 = nn.Linear(self.fourth_layer_parameter_size, self.last_parameter_size)

        self.norm_layer1 = nn.BatchNorm1d(self.first_layer_parameter_size)
        self.norm_layer2 = nn.BatchNorm1d(self.second_layer_parameter_size)
        self.norm_layer3 = nn.BatchNorm1d(self.third_layer_parameter_size)
        self.norm_layer4 = nn.BatchNorm1d(self.fourth_layer_parameter_size)

        self.leakyrelu = nn.LeakyReLU()

        self.binary_layer = nn.Linear(self.last_parameter_size, 1)

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, input_x, output_x):
        batch_size = input_x.shape[0]
        
        if len(input_x.shape) > 3:
            embed_input = self.autoencoder.embedding(input_x.view(batch_size, -1).long())
            embed_output = self.autoencoder.embedding(output_x.view(batch_size, -1).long())
        else:
            embed_input = self.autoencoder.embedding(input_x.view(batch_size, -1).long())
            embed_output = self.autoencoder.embedding(output_x.view(batch_size, -1).long())

        input_feature = self.autoencoder.encoder(embed_input)
        output_feature = self.autoencoder.encoder(embed_output)

        input_mu = self.autoencoder.mu_layer(input_feature)
        input_sigma = self.autoencoder.sigma_layer(input_feature)
        intput_std = torch.exp(0.5 * input_sigma)
        input_eps = torch.randn_like(intput_std)
        input_latent_vector = input_mu + intput_std * input_eps

        output_mu = self.autoencoder.mu_layer(output_feature)
        output_sigma = self.autoencoder.sigma_layer(output_feature)
        output_std = torch.exp(0.5 * output_sigma)
        output_eps = torch.randn_like(output_std)
        output_latent_vector = output_mu + output_std * output_eps

        concat_feature = torch.cat((input_latent_vector, output_latent_vector), dim=2)

        fusion_feature = self.fusion_layer1(concat_feature.view(batch_size, -1))
        fusion_feature = self.norm_layer1(fusion_feature)
        fusion_feature = self.leakyrelu(fusion_feature)

        fusion_feature = self.fusion_layer2(fusion_feature)
        fusion_feature = self.norm_layer2(fusion_feature)
        fusion_feature = self.leakyrelu(fusion_feature)
        
        fusion_feature = self.fusion_layer3(fusion_feature)
        fusion_feature = self.norm_layer3(fusion_feature)
        fusion_feature = self.leakyrelu(fusion_feature)
        
        fusion_feature = self.fusion_layer4(fusion_feature)
        fusion_feature = self.norm_layer4(fusion_feature)
        fusion_feature = self.leakyrelu(fusion_feature)

        fusion_feature = self.fusion_layer5(fusion_feature)
        fusion_feature = self.leakyrelu(fusion_feature)
        
        filter_output = self.binary_layer(fusion_feature)
        filter_output = torch.sigmoid(filter_output)
        output = []
        for i in range(batch_size):
            output.append(filter_output[i])
        output = torch.tensor(output).to('cuda')
        return output
    
train_batch_size = 64
valid_batch_size = 1
lr = 5e-5
batch_size = train_batch_size
epochs = 100
seed = 32
model_name = 'vae'
class_num = 1

new_model = new_idea_vae('/home/jovyan/Desktop/Wongyu/Cross_vae_Linear_origin_b64_lr1e-3_4.pt').to('cuda')
train_dataset_name = '/home/jovyan/Desktop/Wongyu/multiple_data/train_data_for_filter.json'
valid_dataset_name = '/home/jovyan/Desktop/Wongyu/multiple_data/valid_data_for_filter.json'
train_dataset = New_ARCDataset(train_dataset_name)
valid_dataset = New_ARCDataset(valid_dataset_name)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn, drop_last=True, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, collate_fn=collate_fn, drop_last=True, shuffle=True)

optimizer = Lion(new_model.parameters(), lr=lr, weight_decay=1e-2)
criteria = nn.BCELoss().to('cuda')
seed_fix(seed)

for epoch in tqdm(range(epochs)):
    train_total_loss = []
    train_total_acc = 0
    valid_total_loss = []
    valid_total_acc = 0
    train_count = 0
    valid_count = 0
    new_model.train()
    for input, output, task in train_loader:
        train_count += train_batch_size
        input = input.to(torch.float32).to('cuda')
        output = output.to(torch.float32).to('cuda')
        output = new_model(input, output)
        task = task.to(torch.float32).to('cuda')
        loss = criteria(output, task)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_total_loss.append(loss)
        train_total_acc += task.eq(torch.round(output)).sum().item()
        
    print(f'train loss: {sum(train_total_loss) / len(train_total_loss):.6f}')
    print(f'train acc: {train_total_acc * 100 / train_count:.2f}%({train_total_acc}/{train_count})')

    new_model.eval()
    for input, output, task in valid_loader:
        valid_count += valid_batch_size
        input = input.to(torch.float32).to('cuda')
        output = output.to(torch.float32).to('cuda')
        output = new_model(input, output)
        task = task.to(torch.float32).to('cuda')
        loss = criteria(output, task)
        valid_total_loss.append(loss)
        valid_total_acc += task.eq(torch.round(output)).sum().item()

    print(f'valid loss: {sum(valid_total_loss) / len(valid_total_loss):.6f}')
    print(f'valid acc: {valid_total_acc * 100 / valid_count:.2f}%({valid_total_acc}/{valid_count})')