import torch
from torch import optim
import numpy as np
import logging

from DeepRobust.image.attack.base_attack import BaseAttack
from DeepRobust.image.utils import onehot_like, arctanh

import ipdb

class NATTACK(BaseAttack):

    def __init__(self, model, device = 'cuda'):
        super(NATTACK, self).__init__(model, device)
        self.model = model
        self.device = device
    
    def generate(self, **kwargs):
        assert self.parse_params(**kwargs)
        return attack(self.model, self.dataloader, self.classnum, 
                           self.clip_max, self.clip_min, self.epsilon,
                           self.population, self.max_iterations, 
                           self.learning_rate, self.sigma, self.target_or_not)
        assert self.check_type_device(self.dataloader)

    def parse_params(self,
                     dataloader,
                     classnum,
                     target_or_not = False, 
                     clip_max = 1, 
                     clip_min = 0, 
                     epsilon = 0.015,
                     population = 300,
                     max_iterations = 400,
                     learning_rate = 0.008,
                     sigma = 0.1
                     ):
        self.dataloader = dataloader
        self.classnum = classnum
        self.target_or_not = target_or_not
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.epsilon = epsilon
        self.population = population
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.sigma = sigma
        return True

def attack(model, loader, classnum, clip_max, clip_min, epsilon, population, max_iterations, learning_rate, sigma, target_or_not):
    
    logging.basicConfig(format = '%(asctime)s - %(levelname)s: %(message)s')
    logger = logging.getLogger('log_nattack')
    logger.setLevel(logging.DEBUG)
    logger.info('Start attack.')
    
    #initialization
    boxmul = (clip_max - clip_min) / 2.
    boxplus = (clip_max + clip_min) / 2.
    totalImages = 0

    for i, (inputs, targets) in enumerate(loader):
        success = False
        ipdb.set_trace()
        print(inputs.size())
        c = inputs.size(1)
        l = inputs.size(2)
        w = inputs.size(3)
        modify = torch.from_numpy(np.random.randn(1,c, l, w) * 0.001).float()
        
        predict = model.forward(inputs)

        if  predict.argmax(dim = 1, keepdim = True) != targets:
            print('skip the wrong example ', i)
            continue
        totalImages += 1

        if (i > 1):
            break
        for runstep in range(max_iterations):

            # random choose sample
            Nsample = torch.from_numpy(np.random.randn(population, c, l, w)).float()

            modify_try = modify.repeat(population,1,1,1) + sigma * Nsample
            logger.debug('modify_try shape__'+str(modify_try.shape))
            
            # calculate g0(z)
            newimg = arctanh((inputs-boxplus) / boxmul)

            # initialize g(z)
            logger.debug('newimg type__' + str(newimg.type()) + '__')
            logger.debug('modify_try type__' + str(modify_try.type()) + '__')

            inputimg = np.tanh(newimg + modify_try) * boxmul + boxplus

            #pending whether attack is successfull every 10 iterations.
            if runstep % 10 == 0:
                # calculate g(z)
                realinputimg = np.tanh(newimg + modify) * boxmul + boxplus

                # calculate dist in miu space
                realdist = realinputimg - (np.tanh(newimg) * boxmul + boxplus)

                realclipdist = np.clip(realdist, -epsilon, epsilon)
                realclipinput = realclipdist + (np.tanh(newimg) * boxmul + boxplus)
                
                l2real = np.sum((realclipinput - (np.tanh(newimg) * boxmul + boxplus))**2)**0.5
                #l2real =  np.abs(realclipinput - inputs.numpy())
                
                info = 'inputs.shape__' + str(inputs.shape)
                logging.debug(info)
                predict = model.forward(realclipinput)
                #outputsreal = sess.run(real_logits, feed_dict={input_xs: realclipinput.transpose(0,2,3,1)})

                print(np.abs(predict).max())
                print('l2real: '+ str(l2real.max()))
                print(predict)
                
                #pending attack
                if (target_or_not == False):
                    if (np.argmax(predict) != targets) and (np.abs(realclipdist).max() <= epsilon):
                        succImages += 1
                        success = True
                        print('clipimage succImages: '+str(succImages)+'  totalImages: '+str(totalImages))
                        print('lirealsucc: '+str(realclipdist.max()))
                        successlist.append(i)
                        printlist.append(runstep)
                        #break
                        return

            # calculate distance
            dist = inputimg - (np.tanh(newimg) * boxmul + boxplus)
            clipdist = np.clip(dist, -epsilon, epsilon)
            clipinput = (clipdist + (np.tanh(newimg) * boxmul + boxplus)).reshape(npop,3,32,32)
            
            target_onehot = np.zeros((1,classnum))

            target_onehot[0][targets]=1.

            #outputs = sess.run(real_logits, feed_dict={input_xs: clipinput.transpose(0,2,3,1)})
            outputs = model.forward(clipinput)
            target_onehot = target_onehot.repeat(npop,0)

            real = np.log((target_onehot * outputs).sum(1)+1e-30)
            other = np.log(((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0] + 1e-30)

            loss1 = np.clip(real - other, 0.,1000)

            Reward = 0.5 * loss1
            
            #Reward = l2dist

            Reward = - Reward

            A = (Reward - np.mean(Reward)) / (np.std(Reward)+1e-7)

            
            modify = modify + (alpha/(npop*sigma)) * ((np.dot(Nsample.reshape(npop,-1).T, A)).reshape(3,32,32))
        if not success:
            faillist.append(i)
            print('failed:',faillist)
        else:
            print('successed:',successlist)
    
    # print(faillist)
    # success_rate = succImages/float(totalImages)
    # np.savez('runstep',printlist)
    # end_time = time.time()
    # print('all time :', end_time - start_time)
    # print('succc rate', success_rate)
