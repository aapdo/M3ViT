#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from torch.utils.checkpoint import checkpoint

class TamModule(nn.Module):
    def __init__(self, p, tasks, input_channels, norm_cfg = None):
        super(TamModule, self).__init__() 
        self.tasks = tasks 
        self.norm_cfg = norm_cfg

        # layers = {}
        # conv_out = {}
        conv_0 = nn.Conv2d(len(tasks)*input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        _, syncbn_fc_0 = build_norm_layer(self.norm_cfg, input_channels)
        self.layers0 = nn.Sequential(conv_0,syncbn_fc_0)
        conv_1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        _, syncbn_fc_1 = build_norm_layer(self.norm_cfg, input_channels)
        self.layers1 = nn.Sequential(conv_1,syncbn_fc_1)

        conv_2 = nn.Conv2d(len(tasks)*input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        _, syncbn_fc_2 = build_norm_layer(self.norm_cfg, input_channels)
        self.layers2 = nn.Sequential(conv_2,syncbn_fc_2)
        # conv_3 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        # _, syncbn_fc_3 = build_norm_layer(self.norm_cfg, input_channels)
        # self.layers3 = nn.Sequential(conv_3,syncbn_fc_3)


        # encoders = {}
        encoder_0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1)
        _, syncbn_encoder_0 = build_norm_layer(self.norm_cfg, input_channels)
        self.encoder0 = nn.Sequential(encoder_0,syncbn_encoder_0)
        encoder_1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1)
        _, syncbn_encoder_1 = build_norm_layer(self.norm_cfg, input_channels)
        self.encoder1 = nn.Sequential(encoder_1,syncbn_encoder_1)

        decoder_0 = nn.ConvTranspose2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1,output_padding=1)
        _, syncbn_decoder_0 = build_norm_layer(self.norm_cfg, input_channels)
        self.decoder0 = nn.Sequential(decoder_0,syncbn_decoder_0)
        decoder_1 = nn.ConvTranspose2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1,output_padding=1)
        _, syncbn_decoder_1 = build_norm_layer(self.norm_cfg, input_channels)
        self.decoder1 = nn.Sequential(decoder_1,syncbn_decoder_1)

        layers3 = {}
        layers4 = {}
        for task in self.tasks:
            conv_3 = nn.Conv2d(len(tasks)*input_channels, 256, kernel_size=3, stride=1, padding=1)
            _, syncbn_fc_3 = build_norm_layer(self.norm_cfg, 256)
            layers3[task] = nn.Sequential(conv_3,syncbn_fc_3)
            conv_4 = nn.Conv2d(256, p.TASKS.NUM_OUTPUT[task], kernel_size=1, stride=1)
            layers4[task] = nn.Sequential(conv_4)

        self.layers3 = nn.ModuleDict(layers3)
        self.layers4 = nn.ModuleDict(layers4)

    def _block0(self, x):
        """Initial conv block with checkpointing"""
        x = self.layers0(x)
        x = F.relu(x, inplace=False)  # Changed to inplace=False for checkpoint safety
        x = self.layers1(x)
        return torch.sigmoid(x)

    def _block2(self, x):
        """Second conv block with checkpointing"""
        x = self.layers2(x)
        return F.relu(x, inplace=False)  # Changed to inplace=False for checkpoint safety

    def _encoder_block(self, x):
        """Encoder block (2x downsampling) with checkpointing"""
        x = self.encoder0(x)
        x = F.relu(x, inplace=False)  # Changed to inplace=False for checkpoint safety
        x = self.encoder1(x)
        return F.relu(x, inplace=False)  # Changed to inplace=False for checkpoint safety

    def _decoder_block(self, x):
        """Decoder block (2x upsampling) with checkpointing"""
        x = self.decoder0(x)
        x = F.relu(x, inplace=False)  # Changed to inplace=False for checkpoint safety
        x = self.decoder1(x)
        return torch.sigmoid(x)

    def forward(self,deepfeature):
        batch,input_channels,H,W=deepfeature[self.tasks[0]].shape
        featurelist = [deepfeature[t] for t in self.tasks]
        featureinput = torch.stack(featurelist,dim=1).reshape(batch,len(self.tasks)*input_channels,H,W).clone()

        # Apply checkpointing to block0 (layers0 + layers1)
        B = checkpoint(self._block0, featureinput, use_reentrant=False)

        # Gating logic (concat with attention weights)
        if len(self.tasks)==2:
            Fb = torch.cat((deepfeature[self.tasks[0]]*B,deepfeature[self.tasks[1]]*(1-B)),dim=1)
        elif len(self.tasks)==3:
            Fb = torch.cat((deepfeature[self.tasks[0]]*B,deepfeature[self.tasks[1]]*(1-B)/2,deepfeature[self.tasks[2]]*(1-B)/2),dim=1)
        elif len(self.tasks)==4:
            Fb = torch.cat((deepfeature[self.tasks[0]]*B/2,deepfeature[self.tasks[1]]*B/2,deepfeature[self.tasks[2]]*(1-B)/2,deepfeature[self.tasks[3]]*(1-B)/2),dim=1)
        elif len(self.tasks)==5:
            Fb = torch.cat((deepfeature[self.tasks[0]]*B/2,deepfeature[self.tasks[1]]*B/2,deepfeature[self.tasks[2]]*(1-B)/3,deepfeature[self.tasks[3]]*(1-B)/3,deepfeature[self.tasks[4]]*(1-B)/3),dim=1)

        # Apply checkpointing to block2
        Fb = checkpoint(self._block2, Fb, use_reentrant=False)

        # Apply checkpointing to encoder block (2x downsampling)
        Fb = checkpoint(self._encoder_block, Fb, use_reentrant=False)

        # Apply checkpointing to decoder block (2x upsampling)
        M = checkpoint(self._decoder_block, Fb, use_reentrant=False)

        # Final feature aggregation with modulation
        if len(self.tasks)==2:
            Ftam = torch.cat((deepfeature[self.tasks[0]]*(1+M),deepfeature[self.tasks[1]]*(1+M)),dim=1)
        elif len(self.tasks)==3:
            Ftam = torch.cat((deepfeature[self.tasks[0]]*(1+M),deepfeature[self.tasks[1]]*(1+M),deepfeature[self.tasks[2]]*(1+M)),dim=1)
        elif len(self.tasks)==4:
            Ftam = torch.cat((deepfeature[self.tasks[0]]*(1+M),deepfeature[self.tasks[1]]*(1+M),deepfeature[self.tasks[2]]*(1+M),deepfeature[self.tasks[3]]*(1+M)),dim=1)
        elif len(self.tasks)==5:
            Ftam = torch.cat((deepfeature[self.tasks[0]]*(1+M),deepfeature[self.tasks[1]]*(1+M),deepfeature[self.tasks[2]]*(1+M),deepfeature[self.tasks[3]]*(1+M),deepfeature[self.tasks[4]]*(1+M)),dim=1)

        # Apply checkpointing to task-specific heads
        out = {}
        for task in self.tasks:
            def task_head(x, task=task):  # Capture task as default argument
                x = self.layers3[task](x)
                x = F.relu(x, inplace=False)  # Changed to inplace=False for checkpoint safety
                return self.layers4[task](x)

            out[task] = checkpoint(task_head, Ftam, use_reentrant=False)

        return out


class SingleTaskModel(nn.Module):
    """ Single-task baseline model with encoder + decoder """
    def __init__(self, backbone: nn.Module, decoder: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder 
        self.task = task

    def forward(self, x):
        out_size = x.size()[2:]
        out = self.decoder(self.backbone(x))
        return {self.task: F.interpolate(out, out_size, mode='bilinear')}


class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list,p=None):
        super(MultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        self.tasks_id ={}
        id=0
        for task in self.tasks:
            self.tasks_id[task]=id
            id=id+1

        self.tam_level0 = False
        self.tam_level1 = False
        self.tam_level2 = False

        if 'model_kwargs' in p:
            self.tam = p['model_kwargs']['tam']
            self.tam_level0 = True
            self.tam_level1 = True
            self.tam_level2 = True
            if 'tam_level0' in p['model_kwargs']:
                self.tam_level0 = p['model_kwargs']['tam_level0']
            if 'tam_level1' in p['model_kwargs']:
                self.tam_level1 = p['model_kwargs']['tam_level1']
            if 'tam_level2' in p['model_kwargs']:
                self.tam_level2 = p['model_kwargs']['tam_level2']
        else:
            self.tam = False

        print('will consider tam in model',self.tam)
        if 'multi_level' in p:
            self.multi_level = p['multi_level']
        else:
            self.multi_level = False
        print('will consider multi level output in model',self.multi_level)

        if 'multi_gate' in p:
            self.multi_gate = p['multi_gate']
        else:
            self.multi_gate = False
        print('will consider multi gate output in model',self.multi_gate)

        # Add use_cv_loss flag
        if 'use_cv_loss' in p:
            self.use_cv_loss = p['use_cv_loss']
        else:
            self.use_cv_loss = False
        print('will use cv_loss in model',self.use_cv_loss)

        # ckpt backbone always returns (out, cv_losses) tuple
        self.use_checkpointing = p.get('use_checkpointing', False)
        print('will use checkpointing in model',self.use_checkpointing)

        if self.tam:
            if self.tam_level0:
                self.tam_model0 = TamModule(p,self.tasks, 256,norm_cfg = dict(type='SyncBN', requires_grad=True))
            if self.tam_level1:
                self.tam_model1 = TamModule(p,self.tasks, 256,norm_cfg = dict(type='SyncBN', requires_grad=True))
            if self.tam_level2:
                self.tam_model2 = TamModule(p,self.tasks, 256,norm_cfg = dict(type='SyncBN', requires_grad=True))
        
    def forward(self, x, single_task=None, task_id = None, sem=None):
        if task_id is not None:
            assert self.tasks_id[single_task]==task_id
        # print('input shape',x.shape)
        out_size = x.size()[2:]
        cv_losses = []  # Initialize cv_losses

        if not self.multi_gate:
            if task_id is None:
                if sem is None:
                    backbone_out = self.backbone(x)
                else:
                    backbone_out = self.backbone(x, sem=sem)
            else:
                if sem is None:
                    backbone_out = self.backbone(x, task_id=task_id)
                else:
                    backbone_out = self.backbone(x, task_id=task_id, sem=sem)

            # Unpack backbone output
            if isinstance(backbone_out, tuple) and self.use_cv_loss and self.use_checkpointing:
                shared_representation, cv_losses = backbone_out
            else:
                shared_representation = backbone_out
                cv_losses = []
            # print('shared_representation',shared_representation.shape,out_size)
            if self.tam and self.training:
                if self.tam_level0:
                    deepfeature0 = {}
                if self.tam_level1:
                    deepfeature1 = {}
                if self.tam_level2:
                    deepfeature2 = {}
            out = {}
            if single_task is not None:
                task_out = {single_task: F.interpolate(self.decoders[single_task](shared_representation), out_size, mode='bilinear')}
                if cv_losses:
                    return task_out, cv_losses
                else:
                    return task_out
            
            for task in self.tasks:
                if self.tam and self.training:
                    out[task], feature0, feature1, feature2 = self.decoders[task](shared_representation)
                    if self.tam_level0:
                        deepfeature0[task] = feature0
                    if self.tam_level1:
                        deepfeature1[task] = feature1
                    if self.tam_level2:
                        deepfeature2[task] = feature2
                    out[task] = F.interpolate(out[task], out_size, mode='bilinear')
                else:  
                    out[task] = F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear')
            
            if self.tam and self.training:
                if self.tam_level0:
                    x = self.tam_model0(deepfeature0)
                    for task in self.tasks:
                        out['tam_level0_%s' %(task)] = F.interpolate(x[task], out_size, mode='bilinear', align_corners=False)
                if self.tam_level1:
                    x = self.tam_model1(deepfeature1)
                    for task in self.tasks:
                        out['tam_level1_%s' %(task)] = F.interpolate(x[task], out_size, mode='bilinear', align_corners=False)
                if self.tam_level2:
                    x = self.tam_model2(deepfeature2)
                    for task in self.tasks:
                        out['tam_level2_%s' %(task)] = F.interpolate(x[task], out_size, mode='bilinear', align_corners=False)

            # Return tuple only if cv_losses is not empty
            if cv_losses:
                return out, cv_losses
            else:
                return out
        else:
            out = {}
            total_cv_loss = None
            if self.tam:
                if self.tam_level0:
                    deepfeature0 = {}
                if self.tam_level1:
                    deepfeature1 = {}
                if self.tam_level2:
                    deepfeature2 = {}

            for task in self.tasks:
                sem = None if task != 'semseg' else sem
                backbone_out = self.backbone(x, task_id=self.tasks_id[task], sem=sem)

                # Unpack backbone output
                if isinstance(backbone_out, tuple) and self.use_cv_loss and self.use_checkpointing:
                    pertask_representation, task_cv_loss = backbone_out
                    total_cv_loss = task_cv_loss if total_cv_loss is None else (total_cv_loss + task_cv_loss)
                else:
                    pertask_representation = backbone_out
                if self.tam and self.training:
                    out[task], feature0, feature1, feature2 = self.decoders[task](pertask_representation)
                    if self.tam_level0:
                        deepfeature0[task] = feature0
                    if self.tam_level1:
                        deepfeature1[task] = feature1
                    if self.tam_level2:
                        deepfeature2[task] = feature2
                    
                    out[task] = F.interpolate(out[task], out_size, mode='bilinear')
                else:
                    out[task] = F.interpolate(self.decoders[task](pertask_representation), out_size, mode='bilinear')

            if self.tam and self.training:
                if self.tam_level0:
                    x = self.tam_model0(deepfeature0)
                    for task in self.tasks:
                        out['tam_level0_%s' %(task)] = F.interpolate(x[task], out_size, mode='bilinear', align_corners=False)
                if self.tam_level1:
                    x = self.tam_model1(deepfeature1)
                    for task in self.tasks:
                        out['tam_level1_%s' %(task)] = F.interpolate(x[task], out_size, mode='bilinear', align_corners=False)
                if self.tam_level2:
                    x = self.tam_model2(deepfeature2)
                    for task in self.tasks:
                        out['tam_level2_%s' %(task)] = F.interpolate(x[task], out_size, mode='bilinear', align_corners=False)

            # Return tuple if cv_loss is enabled and checkpointing is used
            if self.use_cv_loss and self.use_checkpointing:
                if total_cv_loss is None:
                    total_cv_loss = x.new_tensor(0.0, requires_grad=True)
                return out, total_cv_loss
            else:
                return out

class TokenMultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list,p=None):
        super(TokenMultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        self.tasks_id ={}
        id=0
        for task in self.tasks:
            self.tasks_id[task]=id
            id=id+1

        print('will consider tam in model', False)
        if 'multi_level' in p:
            self.multi_level = p['multi_level']
        else:
            self.multi_level = False
        print('will consider multi level output in model',self.multi_level)

        if 'multi_gate' in p:
            self.multi_gate = p['multi_gate']
        else:
            self.multi_gate = False
        print('will consider multi gate output in model',self.multi_gate)

        # Add use_cv_loss flag
        if 'use_cv_loss' in p:
            self.use_cv_loss = p['use_cv_loss']
        else:
            self.use_cv_loss = False
        print('will use cv_loss in model',self.use_cv_loss)

        # ckpt backbone always returns (out, cv_losses) tuple
        self.use_checkpointing = p.get('use_checkpointing', False)
        print('will use checkpointing in model',self.use_checkpointing)

    def forward(self, x, single_task=None, task_id = None, sem=None):
        if task_id is not None:
            assert self.tasks_id[single_task]==task_id
        # print('input shape',x.shape)
        out_size = x.size()[2:]
        if self.multi_gate:
            # 모든 태스크 한번에 다 실행하고, 블럭 단위로도 모든 태스크 처리하고 다음 블럭. 다음 블럭에서 모든 태스크 처리하고 그 다음. 이렇게 넘어가야함.
            backbone_out = self.backbone(x)

            # Unpack backbone output (task_outputs_dict, total_cv_loss)
            if isinstance(backbone_out, tuple) and self.use_cv_loss and self.use_checkpointing:
                task_outputs_dict, total_cv_loss = backbone_out
            else:
                task_outputs_dict = backbone_out
                total_cv_loss = x.new_tensor(0.0)

            out = {}
            for task in self.tasks:
                out[task] = F.interpolate(self.decoders[task](task_outputs_dict[self.tasks_id[task]]), out_size, mode='bilinear')

            return out, total_cv_loss
        else:
            if task_id is None:
                if sem is None:
                    shared_representation = self.backbone(x)
                else:
                    shared_representation = self.backbone(x, sem=sem)
            else:
                if sem is None:
                    shared_representation = self.backbone(x, task_id=task_id)
                else:
                    shared_representation = self.backbone(x, task_id=task_id, sem=sem)
            # print('shared_representation',shared_representation.shape,out_size)
            out = {}
            if single_task is not None:
                return {single_task: F.interpolate(self.decoders[single_task](shared_representation), out_size, mode='bilinear')}

            for task in self.tasks:
                out[task] = F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear')

            return out
                


class MultiTaskModel_Mixture(nn.Module):
    """ Multi-task baseline model with mixture encoder + task-specific decoders """
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(MultiTaskModel_Mixture, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks

    def forward(self, x, y, overhead=0, prob=0):
        out_size = x.size()[2:]
        shared_representation={task: self.backbone(x,y,task,overhead,prob) for task in self.tasks}
        return {task: F.interpolate(self.decoders[task](shared_representation[task]), out_size, mode='bilinear') for task in self.tasks}