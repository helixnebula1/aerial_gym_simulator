import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, img_dim, action_dim):
        super(DQN, self).__init__()

        self.img_block = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.Linear(128, 16),
            nn.Linear(16, 8)
        )

        self.latest_img_block = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.Linear(128, 16),
            nn.Linear(16, 16)
        )

        self.combine_img_layer = nn.Linear(32, 16)

        self.x_pos_block = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 8)
        )

        self.pos_block = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 4)
        )

        self.combine_pos_layer = nn.Linear(16, 8)

        self.combined_img_pos_block = nn.Sequential(
            nn.Linear(24, 32),
            nn.Linear(32, action_dim)
        )

    def forward(self, depth_imgs, rel_pos):

        i1 = self.img_block(depth_imgs[:, 0, :]) # t-2
        i2 = self.img_block(depth_imgs[:, 1, :]) # t-1
        i3 = self.latest_img_block(depth_imgs[:, 2, :]) # t
        i = torch.cat((i1, i2, i3), dim=1)
        i = self.combine_img_layer(i)

        x = self.x_pos_block(rel_pos[:, 0].unsqueeze(1))
        y = self.pos_block(rel_pos[:, 1].unsqueeze(1))
        z = self.pos_block(rel_pos[:, 2].unsqueeze(1))
        pos = torch.cat((x, y, z), dim=1)
        pos = self.combine_pos_layer(pos)

        combined = torch.cat((i, pos), dim=1)
        combined = self.combined_img_pos_block(combined)


        return combined

    #def output(self, forward?):        # So we can obtain the gradient of the forward.  Can't with softmax in there
        #combined = F.softmax(forward, dim=1)
        #return combined