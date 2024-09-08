from torch.utils.data import Dataset
import numpy as np
import json
import torch
from collections import OrderedDict
# from dataset import New_ARCDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from lion_pytorch import Lion
import wandb
import random
import torch.backends.cudnn as cudnn
import numpy as np
import pdb

class New_ARCDataset(Dataset):
  def __init__(self, file_name, mode=None, permute_mode=False):
    self.dataset = None
    self.mode = mode
    self.count_boundary = 2500
    self.count = 0
    self.permute_mode = permute_mode
    self.use_permute_mode = False # True
    self.permute_color = np.random.choice(11, 11, replace=False)
    with open(file_name, 'r') as f:
        self.dataset = json.load(f)
    if self.mode == 'task':
        task_list = list(set(self.dataset['task']))
        self.task_dict = OrderedDict()
        for i, task in enumerate(task_list):
            self.task_dict[task] = i
    elif 'concept' in mode and 'multi' in mode:
        self.categories = ['AboveBelow', 'Center', 'CleanUp', 'CompleteShape', 'Copy', 'Count', 'ExtendToBoundary', 'ExtractObjects', 'FilledNotFilled', 'HorizontalVertical', 'InsideOutside', 'MoveToBoundary', 'Order', 'SameDifferent', 'TopBottom2D', 'TopBottom3D']
    elif 'multi' in mode:
        self.categories = ['Move', 'Color', 'Object', 'Pattern', 'Count', 'Crop', 'Boundary', 'Center', 'Resize', 'Inside', 'Outside', 'Remove', 'Copy', 'Position', 'Direction', 'Bitwise', 'Connect', 'Order', 'Combine', 'Fill']
  def __len__(self):
    if self.mode == 'Auto_encoder':
        return len(self.dataset['data'])
    else:
        return len(self.dataset['input'])

  def __getitem__(self,idx):
    if self.mode == 'Auto_encoder':
        x = self.dataset['data'][idx]
        size = self.dataset['size'][idx]
        if self.permute_mode:
            self.permute_color = np.random.choice(11, 11, replace=False)
            for i in range(30):
                for j in range(30):
                    x[i][j] = self.permute_color[x[i][j]]
        return torch.tensor(x), torch.tensor(size)
    elif 'multi-bc' in self.mode:
        x = self.dataset['input'][idx]
        y = self.dataset['output'][idx]
        multi_labels = []
        if self.use_permute_mode and self.permute_mode and self.count % self.count_boundary == 0:
            if self.count_boundary > 1:
                self.count_boundary -= 1
            # else:
            #     self.use_permute_mode = False
            self.permute_color = np.random.choice(11, 11, replace=False)
            for i in range(30):
                for j in range(30):
                    x[i][j] = self.permute_color[x[i][j]]
                    y[i][j] = self.permute_color[y[i][j]]
        # for category in self.categories:
        #     temp = [1 if category in self.dataset['task'][idx] else 0]
        # multi_labels.append(temp)
        self.count += 1
        return torch.tensor(x), torch.tensor(y), torch.tensor(self.dataset['task'][idx])
    else:
        x = self.dataset['input'][idx]
        y = self.dataset['output'][idx]
        if self.permute_mode:
            self.permute_color = np.random.choice(11, 11, replace=False)
            for i in range(30):
                for j in range(30):
                    x[i][j] = self.permute_color[x[i][j]]
                    y[i][j] = self.permute_color[y[i][j]]
        task = self.task_dict[self.dataset['task'][idx]]
        return torch.tensor(x), torch.tensor(y), task
def seed_fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
def set_wandb(target_name, mode, seed, optimizer, dataset='ARC'):
    run = wandb.init(project=f'{dataset}_{target_name}_{mode}', entity='whatchang', )
    if mode == 'train':
        config = {
            'learning_rate': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'optimizer': optimizer
        }
        wandb.config.update(config)
        wandb.run.name = f'{model_name}_o{optimizer}_b{batch_size}_e{epochs}_s{seed}_p{permute_mode}'
    wandb.run.save()
    return run

class Autoencoder_batch1_c10(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(10, 20, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            )

        self.proj = nn.Linear(10, 11)

    def forward(self, x):
        feature_map = self.encoder(x)
        output = self.decoder(feature_map)

        return output

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
            embed_x = self.embedding(x.reshape(batch_size, 900).to(torch.long))
        else:
            embed_x = self.embedding(x.reshape(1, 900).to(torch.long))
        feature_map = self.encoder(embed_x)
        mu = self.mu_layer(feature_map)
        sigma = self.sigma_layer(feature_map)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        latent_vector = mu + std * eps
        output = self.decoder(latent_vector)
        output = self.proj(output).reshape(-1,30,30,11)#.permute(0,3,1,2)

        return output
    

class new_idea_vae(nn.Module):
    def __init__(self, model_file):
        super().__init__()

        self.autoencoder = vae_Linear_origin()
        self.autoencoder.load_state_dict(torch.load(model_file))
        self.auto_encoder_freeze()

        self.first_layer_parameter_size = 128
        self.second_layer_parameter_size = 128
        self.third_layer_parameter_size = 64
        self.last_parameter_size = 128
        self.num_categories = 1 #16

        self.binary_layer = nn.Linear(128, 1)

        self.fusion_layer1 = nn.Linear(2*128*900, self.first_layer_parameter_size)
        self.fusion_layer2 = nn.Linear(self.first_layer_parameter_size, self.second_layer_parameter_size)
        self.fusion_layer3 = nn.Linear(self.second_layer_parameter_size, self.third_layer_parameter_size)

        # self.task_proj = nn.Linear(128, 20)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.norm_layer1 = nn.BatchNorm1d(self.first_layer_parameter_size)
        self.norm_layer2 = nn.BatchNorm1d(self.second_layer_parameter_size)
        self.norm_layer3 = nn.BatchNorm1d(self.third_layer_parameter_size)


        #TODO Modulelist와 for문으로 다시 작성하기
        self.move_layer = nn.Linear(self.last_parameter_size, 1)
        self.color_layer = nn.Linear(self.last_parameter_size, 1)
        self.object_layer = nn.Linear(self.last_parameter_size, 1)
        self.pattern_layer = nn.Linear(self.last_parameter_size, 1)
        self.count_layer = nn.Linear(self.last_parameter_size, 1)
        self.crop_layer = nn.Linear(self.last_parameter_size, 1)
        self.boundary_layer = nn.Linear(self.last_parameter_size, 1)
        self.center_layer = nn.Linear(self.last_parameter_size, 1)
        self.resie_layer = nn.Linear(self.last_parameter_size, 1)
        self.inside_layer = nn.Linear(self.last_parameter_size, 1)
        self.outside_layer = nn.Linear(self.last_parameter_size, 1)
        self.remove_layer = nn.Linear(self.last_parameter_size, 1)
        self.copy_layer = nn.Linear(self.last_parameter_size, 1)
        self.position_layer = nn.Linear(self.last_parameter_size, 1)
        self.direction_layer = nn.Linear(self.last_parameter_size, 1)
        self.bitwise_layer = nn.Linear(self.last_parameter_size, 1)
        self.connect_layer = nn.Linear(self.last_parameter_size, 1)
        self.order_layer = nn.Linear(self.last_parameter_size, 1)
        self.combine_layer = nn.Linear(self.last_parameter_size, 1)
        self.fill_layer = nn.Linear(self.last_parameter_size, 1)


        self.AboveBelow_layer = nn.Linear(self.last_parameter_size, 1)
        self.Center_layer = nn.Linear(self.last_parameter_size, 1)
        self.CleanUp_layer = nn.Linear(self.last_parameter_size, 1)
        self.CompleteShape_layer = nn.Linear(self.last_parameter_size, 1)
        self.Copy_layer = nn.Linear(self.last_parameter_size, 1)
        self.Count_layer = nn.Linear(self.last_parameter_size, 1)
        self.ExtendToBoundary_layer = nn.Linear(self.last_parameter_size, 1)
        self.ExtractObjects_layer = nn.Linear(self.last_parameter_size, 1)
        self.FilledNotFilled_layer = nn.Linear(self.last_parameter_size, 1)
        self.HorizontalVertical_layer = nn.Linear(self.last_parameter_size, 1)
        self.InsideOutside_layer = nn.Linear(self.last_parameter_size, 1)
        self.MoveToBoundary_layer = nn.Linear(self.last_parameter_size, 1)
        self.Order_layer = nn.Linear(self.last_parameter_size, 1)
        self.SameDifferent_layer = nn.Linear(self.last_parameter_size, 1)
        self.TopBottom2D_layer = nn.Linear(self.last_parameter_size, 1)
        self.TopBottom3D_layer = nn.Linear(self.last_parameter_size, 1)

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, input_x, output_x):
        batch_size = input_x.shape[0]
        if len(input_x.shape) > 3:
            batch_size = input_x.shape[0]
            embed_input = self.autoencoder.embedding(input_x.reshape(batch_size, 900).to(torch.long))
            embed_output = self.autoencoder.embedding(output_x.reshape(batch_size, 900).to(torch.long))
        else:
            embed_input = self.autoencoder.embedding(input_x.reshape(-1, 900).to(torch.long))
            embed_output = self.autoencoder.embedding(output_x.reshape(-1, 900).to(torch.long))
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

        concat_feature = torch.cat((input_latent_vector, output_latent_vector),dim=2)

        # pdb.set_trace()

        fusion_feature = self.fusion_layer1(concat_feature.reshape(batch_size, -1))
        # fusion_feature = self.dropout(fusion_feature)
        fusion_feature = self.norm_layer1(fusion_feature)
        # fusion_feature = self.relu(fusion_feature)
        fusion_feature = self.leaky_relu(fusion_feature)

        fusion_feature = self.fusion_layer2(fusion_feature)
        # fusion_feature = self.dropout(fusion_feature)
        fusion_feature = self.norm_layer2(fusion_feature)
        # fusion_feature += pre_fusion_feature
        # fusion_feature = self.relu(fusion_feature)
        pre_fusion_feature = self.leaky_relu(fusion_feature)

        fusion_feature = self.fusion_layer2(fusion_feature)
        # fusion_feature = self.dropout(fusion_feature)
        fusion_feature = self.norm_layer2(fusion_feature)
        fusion_feature += pre_fusion_feature
        # fusion_feature = self.relu(fusion_feature)
        fusion_feature = self.leaky_relu(fusion_feature)

        # fusion_feature = self.fusion_layer2(fusion_feature)
        # # fusion_feature = self.dropout(fusion_feature)
        # fusion_feature = self.norm_layer2(fusion_feature)
        # # fusion_feature += pre_fusion_feature
        # # fusion_feature = self.relu(fusion_feature)
        # fusion_feature = self.leaky_relu(fusion_feature)



        # output = self.task_proj(fusion_feature)


        # ===================== 경계선 ======================#


        # move_output = self.move_layer(fusion_feature)
        # color_output = self.color_layer(fusion_feature)
        # object_output = self.object_layer(fusion_feature)
        # pattern_output = self.pattern_layer(fusion_feature)
        # count_output = self.count_layer(fusion_feature)
        # crop_output = self.crop_layer(fusion_feature)
        # boundary_output = self.boundary_layer(fusion_feature)
        # center_output = self.center_layer(fusion_feature)
        # resize_output = self.resie_layer(fusion_feature)
        # inside_output = self.inside_layer(fusion_feature)
        # outside_output = self.outside_layer(fusion_feature)
        # remove_output = self.remove_layer(fusion_feature)
        # copy_output = self.copy_layer(fusion_feature)
        # position_output = self.position_layer(fusion_feature)
        # direction_output = self.direction_layer(fusion_feature)
        # bitwise_output = self.bitwise_layer(fusion_feature)
        # connect_output = self.connect_layer(fusion_feature)
        # order_output = self.order_layer(fusion_feature)
        # combine_output = self.combine_layer(fusion_feature)
        # fill_output = self.fill_layer(fusion_feature)
        #
        # output = torch.stack([move_output, color_output, object_output, pattern_output, count_output, crop_output, boundary_output, center_output, resize_output, inside_output, outside_output, remove_output, copy_output, position_output, direction_output, bitwise_output, connect_output, order_output, combine_output, fill_output])

        # ===================== 경계선 ======================#

        # AboveBelow_output = self.AboveBelow_layer(fusion_feature)
        # Center_output = self.Center_layer(fusion_feature)
        # CleanUp_output = self.CleanUp_layer(fusion_feature)
        # CompleteShape_output = self.CompleteShape_layer(fusion_feature)
        # Copy_output = self.Copy_layer(fusion_feature)
        # Count_layer = self.Count_layer(fusion_feature)
        # ExtendToBoundary_output = self.ExtendToBoundary_layer(fusion_feature)
        # ExtractObjects_output = self.ExtractObjects_layer(fusion_feature)
        # FilledNotFilled_output = self.FilledNotFilled_layer(fusion_feature)
        # HorizontalVertical_output = self.HorizontalVertical_layer(fusion_feature)
        # InsideOutside_output = self.InsideOutside_layer(fusion_feature)
        # MoveToBoundary_output = self.MoveToBoundary_layer(fusion_feature)
        # Order_output = self.Order_layer(fusion_feature)
        # SameDifferent_output = self.SameDifferent_layer(fusion_feature)
        # TopBottom2D_output = self.TopBottom2D_layer(fusion_feature)
        # TopBottom3D_output = self.TopBottom3D_layer(fusion_feature)
        # ===================== 경계선 ======================#

        Filter_output = self.binary_layer(fusion_feature)

        output = Filter_output
        # output = torch.stack([AboveBelow_output, Center_output, CleanUp_output, CompleteShape_output, Copy_output, Count_layer, ExtendToBoundary_output, ExtractObjects_output, FilledNotFilled_output, HorizontalVertical_output, InsideOutside_output, MoveToBoundary_output, Order_output, SameDifferent_output])#, TopBottom2D_output, TopBottom3D_output])

        return output
    

train_batch_size = 32
valid_batch_size = 1
lr = 5e-5
batch_size = train_batch_size
epochs = 100
seed = 777
use_wandb = False
permute_mode = False
use_scheduler = False
model_name = 'vae'
mode = 'concept-multi-bc'
scheduler_name = 'lambda'
lr_lambda = 0.97
class_num = 1 #14 if 'concept' in mode else 20

new_model = new_idea_vae('Cross_vae_Linear_origin_b64_lr1e-3_4.pt').to('cuda')
train_dataset_name = 'Research/data/Train_Concept.json'
valid_dataset_name = 'Research/data/Valid_Concept.json'
# train_dataset_name = 'data/train_new_idea_task_concept_sample2.json'
# valid_dataset_name = 'data/valid_new_idea_task_concept_sample2.json'
# train_dataset_name = 'data/train_new_idea_task_sample2_.json'
# valid_dataset_name = 'data/valid_new_idea_task_sample2_.json'
train_dataset = New_ARCDataset(train_dataset_name, mode=mode, permute_mode=permute_mode)
valid_dataset = New_ARCDataset(valid_dataset_name, mode=mode)
kind_of_dataset = 'Concept_task_sample2' if 'concept' in train_dataset_name else 'ARC_task_sample2' if 'sample2' in train_dataset_name else 'ARC'
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True, shuffle=True)

# optimizer = optim.AdamW(new_model.parameters(), lr=lr)
optimizer = Lion(new_model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: lr_lambda ** epoch)

check_value = torch.tensor([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32).to('cuda')

if 'multi-bc' in mode:
    criteria = nn.BCELoss().to('cuda')
else:
    criteria = nn.CrossEntropyLoss().to('cuda')
seed_fix(777)

if use_wandb:
    set_wandb('new_idea', 'train', seed, 'Lion', kind_of_dataset)

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
        task = task.to(torch.long).to('cuda')
        output = new_model(input, output)

        if 'multi-bc' in mode:
            task_losses = []
            for i in range(task.shape[0]):
                # loss = criteria(nn.functional.softmax(output).permute(1,0,2)[i], task[i].to(torch.float32))
                loss = criteria(nn.functional.sigmoid(output).squeeze(), task.to(torch.float32))
                # task_losses.append(loss)
            # torch.mean(task_losses).backward()
        else:
            loss = criteria(output, task)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_total_loss.append(loss)
        if 'multi-bc' in mode:
            temp_list = []
            for i in range(train_batch_size):
                temp_list.append(torch.where(torch.round(nn.functional.sigmoid(output)[i]).eq(task[i]).sum() == class_num,1, 0))
                if not torch.round(nn.functional.sigmoid(output)[i]).eq(check_value).sum() == class_num:
                    # print(1)
                    pass
            train_total_acc += torch.tensor(temp_list).sum()
            # train_total_acc += output_argmax.permute(1,0,2).eq(task).sum().item()
        else:
            output_softmax= torch.softmax(output,dim=1)
            output_argmax = torch.argmax(output_softmax, dim=1)
            train_total_acc += output_argmax.eq(task).sum().item()
        # if output_argmax == task:
        #     train_total_acc += 1

    if use_scheduler:
        scheduler.step()
    print(f'train loss: {sum(train_total_loss) / len(train_total_loss):.6f}')
    if 'multi' in mode:
        print(f'train acc: {train_total_acc * 100 / train_count:.2f}%({train_total_acc}/{train_count})')
    else:
        print(f'train acc: {train_total_acc * 100 / train_count:.2f}%({train_total_acc}/{train_count})')
    if use_wandb:
        wandb.log({
            "train_loss": sum(train_total_loss) / len(train_total_loss),
            "train_acc": train_total_acc / train_count,
            "train_correct_num": train_total_acc,
        }, step=epoch)

    new_model.eval()
    for input, output, task in valid_loader:
        valid_count += valid_batch_size
        input = input.to(torch.float32).to('cuda')
        output = output.to(torch.float32).to('cuda')
        task = task.to(torch.long).to('cuda')
        output = new_model(input, output)

        if 'multi-bc' in mode:
            loss = criteria(nn.functional.sigmoid(output).squeeze(), task.squeeze().to(torch.float32))
        else:
            loss = criteria(output, task)
        valid_total_loss.append(loss)

        if 'multi-bc' in mode:
            temp_list = []
            for i in range(valid_batch_size):
                temp_list.append(torch.where(torch.round(nn.functional.sigmoid(output)[i]).eq(task[i]).sum() == class_num,1, 0))
            valid_total_acc += torch.tensor(temp_list).sum()
            # valid_total_acc += torch.round(nn.functional.sigmoid(output.permute(1,0,2))).eq(task)
        else:
            output_softmax = torch.softmax(output, dim=1)
            output_argmax = torch.argmax(output_softmax, dim=1)
            valid_total_acc += output_argmax.eq(task).sum().item()
        # if output_argmax == task:
        #     valid_total_acc += 1
    print(f'valid loss: {sum(valid_total_loss) / len(valid_total_loss):.6f}')
    if 'multi' in mode:
        print(f'valid acc: {valid_total_acc*100 / valid_count:.2f}%({valid_total_acc}/{valid_count})')
    else:
        print(f'valid acc: {valid_total_acc*100 / valid_count:.2f}%({valid_total_acc}/{valid_count})')

    if use_wandb:
        wandb.log({
            "valid_loss": sum(valid_total_loss) / len(valid_total_loss),
            "valid_acc": valid_total_acc / valid_count,
            "valid_correct_num": valid_total_acc,
        }, step=epoch)

torch.save(new_model, 'Old_Filter_Model.pt')