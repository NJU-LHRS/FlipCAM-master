import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.label_utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)


#         grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
#         a = grid_image
#         writer.add_image('Image', grid_image, global_step)
#         grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
#                                                        dataset=dataset), 3, normalize=False, range=(0, 255))
#         writer.add_image('Predicted labelsssssssss', grid_image, global_step)
#         grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
#                                                        dataset=dataset), 3, normalize=False, range=(0, 255))
#         b = grid_image
#         writer.add_image('Groundtruth label', grid_image, global_step)
#
#
          # 输出的热力图
        grid_image = make_grid(output[:3,1].clone().cpu().data, 3, normalize=True)
        writer.add_image('foreground heatmap', grid_image, global_step)