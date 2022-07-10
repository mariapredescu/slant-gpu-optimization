import torch
import subjectlist as subl
import os
import argparse
import torchsrc
import time
import torch.nn as nn

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_network(net):
        num_params = 0
        for param in net.parameters():
                num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)

parser = argparse.ArgumentParser()
parser.add_argument('--piece', default='3_3_3', help='1_1_1 | 1_1_3 | 3_3_3 etc.')
parser.add_argument('--model_dir', help='where the model saved')
parser.add_argument('--test_img_dir', help='normalized image dir')
parser.add_argument('--out_dir', help='output dir')
parser.add_argument('--used_epoch', type=int, default=27, help='epoch of finetune model')




opt = parser.parse_args()
print(opt)

# hyper parameters

epoch_num = opt.used_epoch+1
start_epoch = opt.used_epoch
batch_size = 100
lmk_num = 133

model_dir = opt.model_dir
test_img_dir = opt.test_img_dir
out_dir = opt.out_dir

piece = opt.piece

piece_map = {}
piece_map['1_1_1'] = [0,        96,             0,      128,            0,      88]
piece_map['3_1_1'] = [76,       172,            0,      128,            0,      88]
piece_map['1_3_1'] = [0,        96,                     92,     220,            0,      88]
piece_map['3_3_1'] = [76,       172,            92,     220,            0,      88]

piece_map['1_1_3'] = [0,        96,             0,      128,            68,     156]
piece_map['3_1_3'] = [76,       172,            0,      128,            68,     156]
piece_map['1_3_3'] = [0,        96,             92,     220,            68,     156]
piece_map['3_3_3'] = [76,       172,            92,     220,            68,     156]

#middle ones
piece_map['2_1_1'] = [38,       134,            0,      128,            0,      88]
piece_map['2_3_1'] = [38,       134,            92,     220,            0,      88]
piece_map['2_1_3'] = [38,       134,            0,      128,            68,     156]
piece_map['2_3_3'] = [38,       134,            92,     220,            68,     156]

piece_map['1_2_1'] = [0,        96,             46,     174,            0,      88]
piece_map['3_2_1'] = [76,       172,            46,     174,            0,      88]
piece_map['1_2_3'] = [0,        96,             46,     174,            68,     156]
piece_map['3_2_3'] = [76,       172,            46,     174,            68,     156]

piece_map['1_1_2'] = [0,        96,             0,      128,            34,     122]
piece_map['3_1_2'] = [76,       172,            0,      128,            34,     122]
piece_map['1_3_2'] = [0,        96,                     92,     220,            34,     122]
piece_map['3_3_2'] = [76,       172,            92,     220,            34,     122]

piece_map['1_2_2'] = [0,        96,             46,     174,            34,     122]
piece_map['3_2_2'] = [76,       172,            46,     174,            34,     122]
piece_map['2_2_2'] = [38,       134,            46,     174,            34,     122]
piece_map['2_1_2'] = [38,       134,            0,      128,            34,     122]
piece_map['2_3_2'] = [38,       134,            92,     220,            34,     122]
piece_map['2_2_1'] = [38,       134,            46,     174,            0,      88]
piece_map['2_2_3'] = [38,       134,            46,     174,            68,     156]


start = time.time()
#load model
train_root_dir = os.path.join(model_dir, piece,'finetune_out')

out = os.path.join(out_dir,piece,'test_out')
mkdir(out)

test_img_subs,test_img_files = subl.get_sub_list(test_img_dir)
test_dict = {}
test_dict['img_subs'] = test_img_subs
test_dict['img_files'] = test_img_files
print("T_load_model = " + str(time.time() - start) + "s")

# torch.set_num_threads(16)
start = time.time()
# load image
test_set = torchsrc.imgloaders.pytorch_loader_allpiece(test_dict,num_labels=lmk_num,piece=piece,piece_map=piece_map)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
print("T_load_imange = " + str(time.time() - start) + "s")


start = time.time()
# load network
model = torchsrc.models.UNet3D(in_channel=1, n_classes=lmk_num)
print("T_load_network = " + str(time.time() - start) + "s")

# load CUDA
cuda = torch.cuda.is_available()
# cuda = False
# torch.manual_seed(1)
device = torch.device("cuda:0,1")
if cuda:
#       torch.cuda.manual_seed(1)
#       model = model.cuda()
#       print("using GPU")
#       torch.set_num_threads(2)
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        model = nn.DataParallel(model, device_ids = [0,1], output_device = 1)
else:
        torch.set_num_threads(32)
        print("using CPU with one thread")
model.to(device)

start = time.time()
# load trainer
trainer = torchsrc.Trainer(
        cuda=cuda,
        model=model,
        train_loader=[],
        test_loader=test_loader,
        train_root_dir = train_root_dir,
        out=out,
        max_epoch = epoch_num,
        batch_size = batch_size,
        lmk_num = lmk_num,

)
print("T_load_trainer = " + str(time.time() - start) + "s")


print("==start training==")


start = time.time()
start_iteration = 1
trainer.epoch = start_epoch
trainer.iteration = start_iteration
trainer.test_epoch()
print("T_test_epoch = " + str(time.time() - start) + "s")