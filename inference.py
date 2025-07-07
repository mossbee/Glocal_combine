import net
import torch
import os
import numpy as np


adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def get_embedding(model, bgr_tensor_path, device):
    # get embedding from model
    with open(bgr_tensor_path, 'rb') as f:
        bgr_tensor_input = torch.load(f)
    with torch.no_grad():
        bgr_tensor_input = bgr_tensor_input.to(device)
        feature, norm = model(bgr_tensor_input)
    return feature

def get_batch_embedding(model, bgr_tensor_paths, device):
    # get batch embedding from model
    bgr_tensor_list = []
    for bgr_tensor_path in bgr_tensor_paths:
        with open(bgr_tensor_path, 'rb') as f:
            bgr_tensor_input = torch.load(f)
        bgr_tensor_list.append(bgr_tensor_input)
    
    bgr_tensor_batch = torch.stack(bgr_tensor_list).to(device)
    
    with torch.no_grad():
        feature, norm = model(bgr_tensor_batch)
    return feature

if __name__ == '__main__':
    pass