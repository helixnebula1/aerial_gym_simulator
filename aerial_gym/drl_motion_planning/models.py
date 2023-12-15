import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN1a(nn.Module): # Tejas'?

    def __init__(self, img_dim, action_dim):
        super(DQN1a, self).__init__()
        self.img_height = 270
        self.img_width = 480

        self.img_block = nn.Sequential(
            nn.Linear(129600, 128),
            nn.Linear(128, 16),
            nn.Linear(16, 8)
        )

        self.latest_img_block = nn.Sequential(
            nn.Linear(129600, 128),
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


        #i1 = depth_imgs[:,:,0].view(-1, self.img_height * self.img_width)
        #i2 = depth_imgs[:,:,1].view(-1, self.img_height * self.img_width)
        #i3 = depth_imgs[:,:,2].view(-1, self.img_height * self.img_width)
        i1 = torch.flatten(depth_imgs[:,:,0])
        i2 = torch.flatten(depth_imgs[:,:,1])
        i3 = torch.flatten(depth_imgs[:,:,2])

        i1 = self.img_block(i1) # t-2

        i2 = self.img_block(i2) # t-1
        i3 = self.latest_img_block(i3) # t
        i = torch.cat((i1, i2, i3), dim=0)

        i = self.combine_img_layer(i)

        #print("forward ", rel_pos[0][1])
        #x = self.x_pos_block(torch.tensor(rel_pos[0][0]))
        #y = self.pos_block(rel_pos[0][1])
        #z = self.pos_block(rel_pos[0][2])


        x = self.x_pos_block(rel_pos[0][0].unsqueeze(0))
        y = self.pos_block((rel_pos[0][1]).unsqueeze(0))
        z = self.pos_block(rel_pos[0][2].unsqueeze(0))
        pos = torch.cat((x, y, z), dim=0)
        pos = self.combine_pos_layer(pos)

        combined = torch.cat((i, pos), dim=0)
        combined = self.combined_img_pos_block(combined)

        return combined

    def forward2(self, depth_imgs, positions):
        # Flatten and concatenate the depth images
        imgs_flat = depth_imgs.view(-1, self.img_height * self.img_width * self.img_channels)
        imgs_processed = self.img_processing(imgs_flat)

        # Process position information
        pos_x = positions[:,0].view(-1,1)
        pos_y = positions[:,1].view(-1,1)
        pos_z = positions[:,2].view(-1,1)

        pos_x_processed = self.pos_processing_x(pos_x)
        pos_y_processed = self.pos_processing_y(pos_y)
        pos_z_processed = self.pos_processing_z(pos_z)

        # Concatenate processed positions
        pos_combined = torch.cat((pos_x_processed, pos_y_processed, pos_z_processed), dim=1)

        # Combine everything
        combined = torch.cat((imgs_processed, pos_combined), dim=1)

        # Get the final output
        q_values = self.combined(combined)
        return q_values
class DQN(nn.Module): # Tejas'?

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

class DQN2(nn.Module):      # This was Shri's
    def __init__(self, img_height, img_width, action_dim):
    #def __init__(self, img_height, img_width, img_channels, output_size):
        super(DQN, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        #self.img_channels = img_channels
        self.img_channels = 1

        # Image processing layers
        self.img_processing = nn.Sequential(
            nn.Linear(self.img_height * self.img_width, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU()
        )

        # Position processing layers
        self.pos_processing_x = nn.Sequential(
            nn.Linear(1,8),
            nn.ReLU()
        )
        self.pos_processing_y = nn.Sequential(
            nn.Linear(1,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU()
        )
        self.pos_processing_z = nn.Sequential(
            nn.Linear(1,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU()
        )

        # Combined layers
        self.combined = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, depth_imgs, positions):
        # Flatten and concatenate the depth images
        imgs_flat = depth_imgs.view(-1, self.img_height * self.img_width * self.img_channels)
        imgs_processed = self.img_processing(imgs_flat)

        # Process position information
        pos_x = positions[:,0].view(-1,1)
        pos_y = positions[:,1].view(-1,1)
        pos_z = positions[:,2].view(-1,1)

        pos_x_processed = self.pos_processing_x(pos_x)
        pos_y_processed = self.pos_processing_y(pos_y)
        pos_z_processed = self.pos_processing_z(pos_z)

        # Concatenate processed positions
        pos_combined = torch.cat((pos_x_processed, pos_y_processed, pos_z_processed), dim=1)

        # Combine everything
        combined = torch.cat((imgs_processed, pos_combined), dim=1)

        # Get the final output
        q_values = self.combined(combined)
        return q_values

class DQN3(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(DQN3, self).__init__()
        #self.img_height = img_height
        #self.img_width = img_width
        #self.img_channels = 1

        # Image processing layers
        self.img_processing = nn.Sequential(
            nn.Linear(observation_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU()
        )

        # Position processing layers
        self.pos_processing_x = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 8)
        )
        self.pos_processing_y = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 4)
        )
        self.pos_processing_z = nn.Sequential(
            nn.Linear(1,8),
            nn.Linear(8,4)
        )

        # Combined layers
        self.combined = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, depth_imgs, positions):
        # Flatten and concatenate the depth images
        imgs_flat = depth_imgs.view(-1, self.img_height * self.img_width * self.img_channels)
        imgs_processed = self.img_processing(imgs_flat)

        # Process position information
        pos_x = positions[:,0].view(-1,1)
        pos_y = positions[:,1].view(-1,1)
        pos_z = positions[:,2].view(-1,1)

        pos_x_processed = self.pos_processing_x(pos_x)
        pos_y_processed = self.pos_processing_y(pos_y)
        pos_z_processed = self.pos_processing_z(pos_z)

        # Concatenate processed positions
        pos_combined = torch.cat((pos_x_processed, pos_y_processed, pos_z_processed), dim=1)

        # Combine everything
        combined = torch.cat((imgs_processed, pos_combined), dim=1)

        # Get the final output
        q_values = self.combined(combined)
        return q_values

class DQN4(nn.Module): # Tejas w/ mod

    def __init__(self, obs_dim, action_dim):
        super(DQN, self).__init__()

        self.img_block = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Linear(128, 16),
            nn.Linear(16, 8)
        )

        self.latest_img_block = nn.Sequential(
            nn.Linear(obs_dim, 128),
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

class DQN5(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(DQN5, self).__init__()
        #self.img_height = img_height
        #self.img_width = img_width
        #self.img_channels = 1

        # Image processing layers
        self.img_processing = nn.Sequential(
            nn.Linear(observation_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU()
        )

        # Position processing layers
        self.pos_processing_x = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 8)
        )
        self.pos_processing_y = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 4)
        )
        self.pos_processing_z = nn.Sequential(
            nn.Linear(1,8),
            nn.Linear(8,4)
        )

        # Combined layers
        self.combined = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, depth_imgs, positions):
        # Flatten and concatenate the depth images
        imgs_flat = depth_imgs.view(-1, self.img_height * self.img_width * self.img_channels)
        imgs_processed = self.img_processing(imgs_flat)

        # Process position information
        pos_x = positions[:, 0].view(-1, 1)
        pos_y = positions[:, 1].view(-1, 1)
        pos_z = positions[:, 2].view(-1, 1)

        pos_x_processed = self.pos_processing_x(pos_x)
        pos_y_processed = self.pos_processing_y(pos_y)
        pos_z_processed = self.pos_processing_z(pos_z)

        # Concatenate processed positions
        pos_combined = torch.cat((pos_x_processed, pos_y_processed, pos_z_processed), dim=1)

        # Combine everything
        combined = torch.cat((imgs_processed, pos_combined), dim=1)

        # Get the final output
        q_values = self.combined(combined)
        return q_values