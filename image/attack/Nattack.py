import torch
from torch import optim
import numpy as np
import logging

from DeepRobust.image.attack.base_attack import BaseAttack
from DeepRobust.image.utils import onehot_like, arctanh


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
    totalImages = 0
    faillist = []
    successlist = []
    for i, (inputs, targets) in enumerate(loader):
        success = False
        print('attack',i,'picture.')
        #print(inputs.size())
        c = inputs.size(1)
        l = inputs.size(2)
        w = inputs.size(3)
        modify = torch.from_numpy(np.random.randn(1,c, l, w) * 0.001).float()
        
        predict = model.forward(inputs)

        if  predict.argmax(dim = 1, keepdim = True) != targets:
            print('skip the wrong example ', i)
            continue
        totalImages += 1

        # if (i > 1):
        #     break

        for runstep in range(max_iterations):
            #a = input()
            # random choose sample
            Nsample = torch.from_numpy(np.random.randn(population, c, l, w)).float()

            modify_try = modify.repeat(population,1,1,1) + sigma * Nsample
            
            # calculate g0(z)
            g0_z = arctanh((inputs * 2) - 1)
            # print('g0', g0_z)
            # initialize g(z)
            gz = np.tanh(g0_z + modify_try) * 1 / 2 + 1 / 2
            # print('gz', gz)

            #pending whether attack is successfull every 10 iterations.
            if runstep % 10 == 0:
                # a = input()
                # calculate g(z)
                realinputimg = np.tanh(g0_z + modify) * 1 / 2 + 1 / 2
                # print(g0_z)
                # print(modify)
                # calculate dist in miu space
                realdist = realinputimg - (np.tanh(g0_z) * 1/2 + 1/2)

                realclipdist = np.clip(realdist, -epsilon, epsilon).float()
                realclipinput = realclipdist + (np.tanh(g0_z) * 1/2 + 1/2)
                # l2real = np.sum((realclipinput - (np.tanh(g0_z) * 1 / 2 + 1 / 2))**2)**0.5
                #l2real = np.abs(realclipinput - inputs.numpy())
                
                info = 'inputs.shape__' + str(inputs.shape)
                logging.debug(info)
                predict = model.forward(realclipinput)
                # print(predict.argmax(dim = 1, keepdim = True), targets)
                #outputsreal = sess.run(real_logits, feed_dict={input_xs: realclipinput.transpose(0,2,3,1)})
                # print('l2real: '+ str(l2real.max()))
                
                #pending attack
                if (target_or_not == False):
                    if (predict.argmax(dim = 1, keepdim = True) != targets) and (np.abs(realclipdist).max() <= epsilon):
                        succImages += 1
                        success = True
                        print('clipimage succImages: '+str(succImages)+' totalImages: '+str(totalImages))
                        print('lirealsucc: '+str(realclipdist.max()))
                        successlist.append(i)
                        printlist.append(runstep)
                        #break
                        return

            # calculate distance
            dist = gz - (np.tanh(g0_z) * 1 / 2 + 1 / 2)
            clipdist = np.clip(dist, -epsilon, epsilon)
            clipinput = (clipdist + (np.tanh(g0_z) * 1 / 2 + 1 / 2)).reshape(population,1,28,28)
            # print(clipinput.size())
            
            target_onehot = np.zeros((1,classnum))

            target_onehot[0][targets]=1.
            # print('target', target_onehot)
            #outputs = sess.run(real_logits, feed_dict={input_xs: clipinput.transpose(0,2,3,1)})

            clipinput = clipinput.float()
            outputs = model.forward(clipinput)
            # print('output', outputs)
            # print(target_onehot)
            real = (target_onehot * outputs.detach().numpy()).sum(1)
            other = ((1. - target_onehot) * outputs.detach().numpy() - target_onehot * 10000.).max(1)
            # prinst('onehot', target_onehot * outputs.detach().numpy())
            # print('other', (1. - target_onehot) * outputs.detach().numpy() - target_onehot * 10000.)
            # print('other.max', ((1. - target_onehot) * outputs.detach().numpy() - target_onehot * 10000.).max(1))

            loss1 = np.clip(real - other, 0.,1000)
            Reward = 0.5 * loss1
            
            #Reward = l2dist

            Reward = - Reward

            A = (Reward - np.mean(Reward)) / (np.std(Reward)+1e-7)
            #print(A.sum())
            modify = modify + torch.from_numpy((learning_rate/(population*sigma)) * ((np.dot(Nsample.reshape(population,-1).T, A)).reshape(1, 1,28,28)))
            # print(np.shape(modify))
            # print('Nsample', np.shape(Nsample.reshape(population,-1)))
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
