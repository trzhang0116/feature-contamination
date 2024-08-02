import torch
import numpy as np
import cv2
from architectures import ARCHITECTURES, get_architecture
from torchvision import models

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
    def close(self):
        self.hook.remove()

def np2tensor(image,dtype):
    "Convert np.array (sz,sz,3) to tensor (1,3,sz,sz), imagenet normalized"

    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    
    #Imagenet norm
    mean=np.array([0.485, 0.456, 0.406])[...,np.newaxis,np.newaxis]
    std = np.array([0.229, 0.224, 0.225])[...,np.newaxis,np.newaxis]
    a = (a-mean)/std
    a = np.expand_dims(a,0)
    return torch.from_numpy(a.astype(dtype, copy=False) )

def tensor2np(img_tensor):
    "Convert tensor (1,3,sz,sz) back to np.array (sz,sz,3), imagenet DEnormalized"
    a = np.squeeze(img_tensor.cpu().detach().numpy())
    mean=np.array([0.485, 0.456, 0.406])[...,np.newaxis,np.newaxis]
    std = np.array([0.229, 0.224, 0.225])[...,np.newaxis,np.newaxis]
    a = a*std + mean
    return np.transpose(a, (1,2,0))


class FilterVisualizer():
    def __init__(self,model):
        self.model = model

    def visualize(self, sz, layer, filter, upscaling_steps=12, upscaling_factor=1.2, lr=0.01, opt_steps=50, blur=5, print_losses=False):
        
        img = (np.random.random((sz,sz, 3)) * 20 + 128.)/255 # value b/t 0 and 1
        # img = (torch.randn((sz,sz, 3)) * 20 + 128.)/255        
        activations = SaveFeatures(layer)  # register hook
        torch.cuda.empty_cache()
        for i in range(upscaling_steps):  
            # convert np to tensor + channel first + new axis, and apply imagenet norm
            img_tensor = np2tensor(img,np.float32)
            img_tensor = img_tensor.cuda()
            img_tensor.requires_grad_()
            if not img_tensor.grad is None:
                img_tensor.grad.zero_(); 
            
            
            optimizer = torch.optim.Adam([img_tensor], lr=lr, weight_decay=0)
            if i > upscaling_steps/2:
                opt_steps_ = int(opt_steps*1.3)
            else:
                opt_steps_ = opt_steps
            for n in range(opt_steps_):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
            
                _=self.model(img_tensor)
                
                incent = torch.full_like(img_tensor, 0.5)

                # print(incent)
                # print(img_tensor)
                loss = -1*activations.features[0, filter].mean() + 1e-2 * torch.norm(img_tensor - incent, p=2)
                if print_losses:
                    if i%3==0 and n%5==0:
                        # print(f'{i} - {n} - {float(loss)}')
                        if loss == 0:
                            break
                loss.backward()
                optimizer.step()
            
            # convert tensor back to np
            img = tensor2np(img_tensor)
            self.output = img
            sz = int(sz)  # calculate new image size
            # print(f'Upscale img to: {sz}')
            img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)  # scale image up
            if blur is not None: img = cv2.blur(img,(blur,blur))  # blur image to reduce high frequency patterns
                
        activations.close()
        return self.output #np.clip(self.output, 0, 1)
    
    def get_transformed_img(self,img,sz):
        '''
        Scale up/down img to sz. Channel last (same as input)
        image: np.array [sz,sz,3], already divided by 255"    
        '''
        return cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)
    
    def most_activated(self, img, layer):
        '''
        image: np.array [sz,sz,3], already divided by 255"    
        '''
        img = cv2.resize(img, (224,224), interpolation = cv2.INTER_CUBIC)
        activations = SaveFeatures(layer)
        img_tensor = np2tensor(img,np.float32)
        img_tensor = img_tensor.cuda()
        
        _=self.model(img_tensor)
        mean_act = [np.squeeze((activations.features[0,i].mean()).numpy()) for i in range(activations.features.shape[1])]
        activations.close()
        return mean_act
    
            
if __name__ == "__main__":

    checkpoint = torch.load("Pre-trained models/checkpoint_resnet32_250epoch.pth.tar")
    model = get_architecture(checkpoint["arch"], 'cifar10')
    model.load_state_dict(checkpoint['state_dict'])
    
    
    m = model.cuda().eval()
    fv = FilterVisualizer(m)
    mods = [*m.children()]
    
    fil = model[1].layer3[4].conv2
    for i in range(64):
        img = fv.visualize(32,fil,i, upscaling_steps=10 ,lr=0.05, print_losses=True, opt_steps=200)

        img = (img - img.min()) / (img.max() - img.min())

        torch.cuda.empty_cache()
        print(i)
        cv2.imwrite("jpg_temp/layer31_{:d}.jpg".format(i), (img*255).astype(int))
