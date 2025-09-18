import argparse
import os
import random
import sys

sys.path.append('mobilevit_v3/')
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
import tensorboardX
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda import amp
from dataset.dataset import get_loader
from train_utils.train_utils import adjust_lr, AvgMeter, save_checkpoint
from validation import validation
from torchvision.utils import make_grid
from configs import config, ckpt_path
from models.model import HASNet
# 我是懒狗
os.environ['TORCH_HOME'] = '../saved_models'
random.seed(118)
np.random.seed(118)
torch.manual_seed(118)
torch.cuda.manual_seed(118)
torch.cuda.manual_seed_all(118)
print('Generator Learning Rate: {}'.format(config['lr']))

model_name = 'HASNet'

# build models
generator = globals()[model_name]()
generator.cuda()

print(generator)
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, config['lr'])


train_loader = get_loader(config)
total_step = len(train_loader)

# size_rates = [0.75,1,1.25]  # multi-scale training
size_rates = [1]  # multi-scale training
use_fp16 = True
scaler = amp.GradScaler(enabled=use_fp16)


# 只用 wbce
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def get_loss(side_out1, side_out2, side_out3, side_out4, target1, target2, target3, target4):
    loss1 = structure_loss(side_out1, target1)
    loss2 = structure_loss(side_out2, target2)
    loss3 = structure_loss(side_out3, target3)
    loss4 = structure_loss(side_out4, target4)
    # sml = get_saliency_smoothness(torch.sigmoid(side_out1), mask1)
    # 1 ，0.9，0.8，0.7 0.8
    return loss1, loss2, loss3, loss4


if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

sw = tensorboardX.SummaryWriter('./results/tensorboard_log/' + model_name + '_log')


def train():
    min_mae = 1
    validation_step = 0
    global_step = 0
    for epoch in range(1, config['epochs'] + 1):
        generator.train()
        loss_record = AvgMeter()
        print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
        for i, pack in enumerate(train_loader, start=1):
            generator_optimizer.zero_grad()
            images = pack['sal_image']
            gts = pack['sal_label']
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()
            b, c, h, w = gts.size()
            target_1 = F.interpolate(gts, size=h // 4, mode='nearest')
            target_2 = F.interpolate(gts, size=h // 8, mode='nearest')
            target_3 = F.interpolate(gts, size=h // 16, mode='nearest')
            # target_4 = F.interpolate(gts, size=h // 32, mode='nearest')
            # torch.autograd.set_detect_anomaly(True)
            with amp.autocast(enabled=use_fp16):
                side_out4, side_out3, side_out2, side_out1 = generator(images)  # hed
                loss1, loss2, loss3, loss4 = get_loss(side_out1, side_out2, side_out3, side_out4, gts,
                                                      target_1,
                                                      target_2, target_3)
                loss = loss1 + loss2 + loss3 + loss4
            generator_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()
            loss_record.update(loss.data, config['batch_size'])
            global_step += 1
            sw.add_scalar('lr', generator_optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss',
                           {'loss_sum': loss.item(), 'loss1': loss1.item(), 'loss2': loss2.item(),
                            'loss3': loss3.item(),
                            'loss4': loss4.item()},
                           global_step=global_step)

            if i % 10 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                      format(datetime.now(), epoch, config['epochs'], i, total_step, loss_record.show()))
            if i % 100 == 0 or i == total_step or i == 1:
                res = side_out1[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = torch.tensor(res).unsqueeze(dim=0)
                show_gt = gts[0].clone().cpu().data
                show_gt = torch.cat((show_gt, show_gt, show_gt), dim=0)
                show_res = torch.cat((res, res, res), dim=0)
                grid_image = make_grid(
                    [images[0].clone().cpu().data, show_gt, show_res], 3,
                    normalize=True)
                sw.add_image('res', grid_image, i)
        adjust_lr(generator_optimizer, config['lr'], epoch, config['decay_rate'], config['decay_epoch'])
        if epoch % 10 == 0 or epoch % config['epochs'] == 0:
            mae, validation_step = validation(generator, config['test_size'], sw, validation_step)
            print('this validation mae is', mae, 'epoch is', epoch)
            sw.add_scalars('mae', {'mae': mae}, global_step=epoch)
            current_mae_avg = mae
            # if current_mae_avg < min_mae and epoch > 150:
            # if current_mae_avg < min_mae and epoch > 150:
            #     min_mae = current_mae_avg
            #     print('the lower mae and saving models...')
            #     path = save_path + model_name + '_%d' % epoch + '_gen.pth'
            #     torch.save(generator.state_dict(), path)
        if epoch % config['epochs'] == 0 or epoch % config['epochs'] == 70:
            path = ckpt_path
            path = path + model_name + '_%d' % epoch + '_gen.pth'
            # if opt.save_opt:
            #     save_checkpoint(path, generator, epoch, generator_optimizer)
            # else:
            torch.save(generator.state_dict(), path)


if __name__ == '__main__':
    train()
    sw.close()
