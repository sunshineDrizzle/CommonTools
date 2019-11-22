import torch

from torch import nn


def _get_truncate_layers(model_frame, indices):
    """
    Subfunction of dnn_truncate to access truncated model recursively.
    """
    if len(indices) > 1:
        parent_sequential = nn.Sequential(*model_frame[:indices[0]].children())
        append_sequential = nn.Sequential(*model_frame[indices[0]].children())
        truncate_model = nn.Sequential(*parent_sequential, _get_truncate_layers(append_sequential, indices[1:]))
    else:
        truncate_model = nn.Sequential(*model_frame[:(indices[0]+1)])
    return truncate_model


def dnn_truncate(netloader, layer):
    """
    Truncate the neural network at the specified convolution layer.
    Notice that all truncated models were consisted of Sequential, which may differ from the orginal model.

    Parameters:
    -----------
    netloader[NetLoader]: a neural network netloader, initialized from NetLoader in io module
    layer[str]: truncated layer.

    Returns:
    --------
    truncated_net[torch.nn.Sequential]: truncated model.
    """
    assert netloader.model is not None, "Please define netloader by calling NetLoader from module io"
    assert netloader.layer2indices is not None, "Please define netloader by calling NetLoader from module io"
    indices = netloader.layer2indices[layer]
    prefc_indices = netloader.layer2indices['prefc']
    model_frame = nn.Sequential(*netloader.model.children())
    truncate_model = _get_truncate_layers(model_frame, indices)
    if 'fc' in layer:
        # Re-define forward method
        def forward(x):
            x = truncate_model[:prefc_indices[0]](x)
            x = torch.flatten(x, 1)
            x = truncate_model[prefc_indices[0]:](x)
            return x
        truncate_model.forward = forward
    return truncate_model


class TransferredNet(nn.Module):
    def __init__(self, truncated_net, fc_in_num, fc_out_num, channel=None, feature_extract=True):
        """
        Connect the truncated_net to a full connection layer.

        Parameters:
        -----------
        truncated_net[torch.nn.Module]: a truncated neural network from the pretrained network
        fc_in_num[int]: the number of the in_features of the full connection layer
        fc_out_num[int]: the number of the out_features of the full connection layer
        channel[iterator]: The indices of out_channels of the selected convolution layer
        feature_extract[bool]: If feature_extract = False, the model is finetuned and all model parameters are updated.
            If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
            https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        """
        super(TransferredNet, self).__init__()
        self.truncated_net = truncated_net
        if feature_extract:
            for param in self.truncated_net.parameters():
                param.requires_grad = False
        self.fc = nn.Linear(fc_in_num, fc_out_num)
        self.channel = channel

    def forward(self, x):
        x = self.truncated_net(x)
        if self.channel is not None:
            # extract the specified channel's output
            x = x[:, self.channel]
        x = x.view(x.size(0), -1)  # (batch_num, unit_num)
        x = self.fc(x)
        return x


# GraftLayer is Taicheng Huang's version of TransferredNet
class GraftLayer(nn.Module):
    def __init__(self, netloader, layer, fc_out_num, channel=None, feature_extract=True):
        """
        Connect the truncated_net to a full connection layer.

        Parameters:
        -----------
        netloader[NetLoader]: a neural network netloader, initialized from NetLoader in io module.
        layer[str]: truncated layer
        fc_out_num[int]: the number of the out_features of the full connection layer
        channel[iterator]: The indices of out_channels of the selected convolution layer
        feature_extract[bool]: If feature_extract = False, the model is finetuned and all model parameters are updated.
            If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
            https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        """
        super(GraftLayer, self).__init__()
        self.truncate_net = dnn_truncate(netloader, layer)
        if feature_extract:
            for param in self.truncate_net.parameters():
                param.requires_grad = False

        # Set a test data to get output size of the truncate_net
        indicator_val = torch.randn((1,3,netloader.img_size[0],netloader.img_size[1]))
        indicator_output = self.truncate_net(indicator_val)

        if channel is not None:
            assert 'conv' in layer, "Selected channel only happened in convolution layer."
        self.channel = channel
        # fc input number
        fc_in_num = indicator_output.view(indicator_output.size(0),-1).shape[-1]
        self.fc = nn.Linear(fc_in_num, fc_out_num)

    def forward(self, x):
        x = self.truncate_net(x)
        # if self.channel is not None :
        # I'm not quite sure how to trained a model with some channels but not all.
        # Therefore channel is not be finished.
        # The original one is
        # if self.channel is not None:
        #   x = x[:, self.channel]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
