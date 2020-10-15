import torch

def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr

class BNOptimizer():

    @staticmethod
    def updateBN(sr_flag, module_list, s, prune_idx, epoch, idx2mask=None, opt=None):
        if sr_flag:
            # s = s if epoch <= opt.epochs * 0.5 else s * 0.01
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                bn_module = module_list[idx][1]
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1
            if idx2mask:
                for idx in idx2mask:
                    bn_module = module_list[idx][1]
                    #bn_module.weight.grad.data.add_(0.5 * s * torch.sign(bn_module.weight.data) * (1 - idx2mask[idx].cuda()))
                    bn_module.weight.grad.data.sub_(0.99 * s * torch.sign(bn_module.weight.data) * idx2mask[idx].cuda())

def parse_module_defs(module_defs):

    CBL_idx = []
    Conv_idx = []
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            if module_defs[i+1]['type'] == 'maxpool' and module_defs[i+2]['type'] == 'route':
                #spp前一个CBL不剪 区分tiny
                ignore_idx.add(i)
            if module_defs[i+1]['type'] == 'route' and 'groups' in module_defs[i+1]:
                ignore_idx.add(i)
            if module_defs[i+1]['type'] == 'convolutional_nobias':
                ignore_idx.add(i)
        elif module_def['type'] == 'convolutional_noconv':
            CBL_idx.append(i)
            ignore_idx.add(i)
        elif module_def['type'] == 'shortcut':
            ignore_idx.add(i-1)
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)

        elif module_def['type'] == 'upsample':
            #上采样层前的卷积层不裁剪
            ignore_idx.add(i - 1)


    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx


def parse_module_defs2(module_defs):
    CBL_idx = []
    Conv_idx = []
    shortcut_idx = dict()
    shortcut_all = set()
    ignore_idx = set()
    CBL_name = []
    all_name=[]
    shortcut_3conv=set()
    for name, parameters in module_defs.named_parameters():
        # print('name: {}, param: {}'.format(name, parameters))
        # print('name: {}'.format(name))
        all_name.append(name)
    for i,name in enumerate(all_name):
        if 'conv' == name.split('.')[-2]:
            if all_name[i+1]==name.replace('.conv.weight','.bn.weight'):
                CBL_idx.append(i)
        elif 'weight' == name.split('.')[-1] and 'bn' != name.split('.')[-2]:
            Conv_idx.append(i)
        elif '8.cv1.conv.weight' == name:#SPP
            ignore_idx.add(i)
        elif '10.conv.weight' == name or '14.conv.weight' == name:  # upsample
            ignore_idx.add(i)
        elif '2.cv1.conv.weight' == name or '4.cv1.conv.weight' == name or '6.cv1.conv.weight' == name:
            shortcut_3conv.add(i)
        elif name.split('.')[0] in ['2','4','6'] and name.split('.')[1]=='m' and name.split('.')[3:]==['cv2','conv','weight']:
            # if name.split('.')[0]=='2' and name.split('.')[2]=='0':
            #     shortcut_idx[i]=shortcut_3conv[0]
            #     shortcut_all.add(shortcut_3conv[0])
            # if name.split('.')[0] == '4' and name.split('.')[2] == '0':
            #     shortcut_idx[i] = shortcut_3conv[1]
            #     shortcut_all.add(shortcut_3conv[1])
            # if name.split('.')[0]=='6' and name.split('.')[2]=='0':
            #     shortcut_idx[i]=shortcut_3conv[2]
            #     shortcut_all.add(shortcut_3conv[2])
            if name.split('.')[2]=='0': #shortcut
                identity_idx=i-13
                shortcut_idx[i]=identity_idx
                shortcut_all.add(identity_idx)
            else:
                identity_idx=i-6
                shortcut_idx[i] = identity_idx
                shortcut_all.add(identity_idx)
            shortcut_all.add(i)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all

    # for module in module_defs.modules():
    #     print(module)

    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            if module_defs[i + 1]['type'] == 'maxpool' and module_defs[i + 2]['type'] == 'route':
                # spp前一个CBL不剪 区分spp和tiny
                ignore_idx.add(i)
            if module_defs[i + 1]['type'] == 'route' and 'groups' in module_defs[i + 1]:
                ignore_idx.add(i)

        elif module_def['type'] == 'upsample':
            # 上采样层前的卷积层不裁剪
            ignore_idx.add(i - 1)

        elif module_def['type'] == 'shortcut':
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':

                # ignore_idx.add(identity_idx)
                shortcut_idx[i - 1] = identity_idx
                shortcut_all.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':

                # ignore_idx.add(identity_idx - 1)
                shortcut_idx[i - 1] = identity_idx - 1
                shortcut_all.add(identity_idx - 1)
            shortcut_all.add(i - 1)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all

def gather_bn_weights(module_list, prune_idx):

    size_list = [module_list[idx][1].weight.data.shape[0] if type(module_list[idx][1]).__name__ is 'BatchNorm2d' else module_list[idx][0].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone() if type(module_list[idx][1]).__name__ is 'BatchNorm2d' else module_list[idx][0].weight.data.abs().clone()
        index += size

    return bn_weights


def obtain_bn_mask(bn_module_weight, thre):

    thre = thre.cuda()
    mask = bn_module_weight.data.abs().ge(thre).float()

    return mask

from copy import deepcopy
import torch.nn.functional as F

def update_activation(i, pruned_model, activation, CBL_idx):
    next_idx = i + 1
    if pruned_model.module_defs[next_idx]['type'] == 'convolutional':
        next_conv = pruned_model.module_list[next_idx][0]
        conv_sum = next_conv.weight.data.sum(dim=(2, 3))
        offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
        if next_idx in CBL_idx:
            next_bn = pruned_model.module_list[next_idx][1]
            next_bn.running_mean.data.sub_(offset)
        else:
            next_conv.bias.data.add_(offset)

def prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask):
    pruned_model = deepcopy(model)
    activations = []
    all_name = []
    all_parameters = []
    activations_dict = {}
    activations_shortcut_dict = {}
    for name, parameters in pruned_model.model.named_parameters():
        # print('name: {}, param: {}'.format(name, parameters))
        print('name: {}'.format(name))
        all_name.append(name)
        all_parameters.append(parameters)
    for i, name in enumerate(all_name):
        if 'conv' == name.split('.')[-2]:  #convolutional
            if all_name[i + 1] == name.replace('.conv.weight', '.bn.weight') and name[2:]!='cv1.conv.weight':
                activation=torch.zeros(all_parameters[0].shape[0]).cuda()
                if i in prune_idx:
                    mask = torch.from_numpy(CBLidx2mask[i]).cuda()
                    bn_module_weight = all_parameters[i + 1]
                    bn_module_weight.data.mul_(mask)
                    bn_module_bias = all_parameters[i + 2]
                    #leaky
                    activation = F.leaky_relu((1 - mask) * bn_module_bias.data, 0.1)
                    #update_activation
                    next_idx = i + 3
                    if all_name[next_idx].split('.')[-2] == 'conv' and all_name[i + 1] == name.replace('.conv.weight', '.bn.weight'):
                        next_conv_weight = all_parameters[next_idx]
                        conv_sum = next_conv_weight.data.sum(dim=(2, 3))
                        offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
                        if next_idx in CBL_idx:
                            # next_bn = pruned_model.module_list[next_idx][1]
                            next_bn_running_mean=pruned_model.model.state_dict()[all_name[next_idx].replace('conv.weight', 'bn.running_mean')]
                            next_bn_running_mean.data.sub_(offset)
                        else:
                            next_conv.bias.data.add_(offset)

                    bn_module_bias.data.mul_(mask)
                    activations_dict[name]=activation
        elif name[2:]=='cv1.conv.weight':
            if i in prune_idx:
                mask = torch.from_numpy(CBLidx2mask[i]).cuda()
                bn_module_weight = all_parameters[i + 1]
                bn_module_weight.data.mul_(mask)
                bn_module_bias = all_parameters[i + 2]
                # leaky
                activation = F.leaky_relu((1 - mask) * bn_module_bias.data, 0.1)
                # update_activation
                next_idx = i + 10
                if all_name[next_idx].split('.')[-2] == 'conv' and all_name[i + 1] == name.replace('.conv.weight',
                                                                                                   '.bn.weight'):
                    next_conv_weight = all_parameters[next_idx]
                    conv_sum = next_conv_weight.data.sum(dim=(2, 3))
                    offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
                    if next_idx in CBL_idx:
                        # next_bn = pruned_model.module_list[next_idx][1]
                        next_bn_running_mean = pruned_model.model.state_dict()[
                            all_name[next_idx].replace('conv.weight', 'bn.running_mean')]
                        next_bn_running_mean.data.sub_(offset)
                    else:
                        next_conv.bias.data.add_(offset)
                bn_module_bias.data.mul_(mask)
                activations_dict[name] = activation
        elif name == '2.m.0.cv2.conv.weight':
            mask = torch.from_numpy(CBLidx2mask[i]).cuda()
            bn_module_weight = all_parameters[i + 1]
            bn_module_weight.data.mul_(mask)
            bn_module_bias = all_parameters[i + 2]
            # leaky
            activation = F.leaky_relu((1 - mask) * bn_module_bias.data, 0.1)
            bn_module_bias.data.mul_(mask)
            activations_dict[name] = activation
            actv1 = activation
            actv2 = activations_dict['2.cv1.conv.weight']
            activation = actv1 + actv2
            activations_shortcut_dict[name] = activation
            # update_activation
            next_conv_weight_cv2 = pruned_model.model.state_dict()['2.cv2.weight']
            next_conv_weight_cv3 = pruned_model.model.state_dict()['2.cv3.weight']
            next_conv_weight = torch.cat(next_conv_weight_cv2, next_conv_weight_cv3)
            conv_sum = next_conv_weight.data.sum(dim=(2, 3))
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            next_bn_running_mean = pruned_model.model.state_dict()['2.bn.running_mean']
            next_bn_running_mean.data.sub_(offset)

        elif name=='4.m.1.cv1.conv.weight':
            actv1=activations_dict['4.m.0.cv2.conv.weight']
            actv2 = activations_dict['4.cv1.conv.weight']
            activation = actv1 + actv2

            next_idx = i + 3
            if all_name[next_idx].split('.')[-2] == 'conv' and all_name[i + 1] == name.replace('.conv.weight','.bn.weight'):
                next_conv_weight = all_parameters[next_idx]
                conv_sum = next_conv_weight.data.sum(dim=(2, 3))
                offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
                if next_idx in CBL_idx:
                    # next_bn = pruned_model.module_list[next_idx][1]
                    next_bn_running_mean = pruned_model.model.state_dict()[
                        all_name[next_idx].replace('conv.weight', 'bn.running_mean')]
                    next_bn_running_mean.data.sub_(offset)
                else:
                    next_conv.bias.data.add_(offset)

            activations_dict[name] = activation
        elif name == '4.m.2.cv1.conv.weight':
            actv1 = activations_dict['4.m.1.cv2.conv.weight']
            actv2 = activations_dict['4.m.1.cv1.conv.weight']
            activation = actv1 + actv2

            next_idx = i + 3
            if all_name[next_idx].split('.')[-2] == 'conv' and all_name[i + 1] == name.replace('.conv.weight',
                                                                                               '.bn.weight'):
                next_conv_weight = all_parameters[next_idx]
                conv_sum = next_conv_weight.data.sum(dim=(2, 3))
                offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
                if next_idx in CBL_idx:
                    # next_bn = pruned_model.module_list[next_idx][1]
                    next_bn_running_mean = pruned_model.model.state_dict()[
                        all_name[next_idx].replace('conv.weight', 'bn.running_mean')]
                    next_bn_running_mean.data.sub_(offset)
                else:
                    next_conv.bias.data.add_(offset)

            activations_dict[name] = activation


    for i, model_def in enumerate(model.module_defs):

        if model_def['type'] == 'convolutional':
            activation = torch.zeros(int(model_def['filters'])).cuda()
            if i in prune_idx:
                mask = torch.from_numpy(CBLidx2mask[i]).cuda()
                bn_module = pruned_model.module_list[i][1]
                bn_module.weight.data.mul_(mask)
                if model_def['activation'] == 'leaky':
                    activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)
                elif model_def['activation'] == 'mish':
                    activation = (1 - mask) * bn_module.bias.data.mul(F.softplus(bn_module.bias.data).tanh())
                update_activation(i, pruned_model, activation, CBL_idx)
                bn_module.bias.data.mul_(mask)
            activations.append(activation)

        elif model_def['type'] == 'shortcut':
            actv1 = activations[i - 1]
            from_layer = int(model_def['from'])
            actv2 = activations[i + from_layer]
            activation = actv1 + actv2
            update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)



        elif model_def['type'] == 'route':
            # spp不参与剪枝，其中的route不用更新，仅占位
            from_layers = [int(s) for s in model_def['layers'].split(',')]
            activation = None
            if len(from_layers) == 1:
                activation = activations[i + from_layers[0] if from_layers[0] < 0 else from_layers[0]]
                if 'groups' in model_def:
                    activation = activation[(activation.shape[0] // 2):]
                update_activation(i, pruned_model, activation, CBL_idx)
            elif len(from_layers) == 2:
                actv1 = activations[i + from_layers[0]]
                actv2 = activations[i + from_layers[1] if from_layers[1] < 0 else from_layers[1]]
                activation = torch.cat((actv1, actv2))
                update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)

        elif model_def['type'] == 'upsample':
            # activation = torch.zeros(int(model.module_defs[i - 1]['filters'])).cuda()
            activations.append(activations[i - 1])

        elif model_def['type'] == 'yolo':
            activations.append(None)

        elif model_def['type'] == 'maxpool':  # 区分spp和tiny
            if model.module_defs[i + 1]['type'] == 'route':
                activations.append(None)
            else:
                activation = activations[i - 1]
                update_activation(i, pruned_model, activation, CBL_idx)
                activations.append(activation)

    return pruned_model