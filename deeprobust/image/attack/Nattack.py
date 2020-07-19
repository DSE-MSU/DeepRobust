import torch
from torch import optim
import numpy as np
import logging

from deeprobust.image.attack.base_attack import BaseAttack
from deeprobust.image.utils import onehot_like, arctanh


class NATTACK(BaseAttack):
    """
    Nattack is a black box attack algorithm. 
    """


    def __init__(self, model, device = 'cuda'):
        super(NATTACK, self).__init__(model, device)
        self.model = model
        self.device = device

    def generate(self, **kwargs):
        """
        Call this function to generate adversarial examples.

        Parameters
        ----------
        kwargs :
            user defined paremeters
        """
 
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
                     epsilon = 0.2,
                     population = 300,
                     max_iterations = 400,
                     learning_rate = 2,
                     sigma = 0.1
                     ):
        """parse_params.

        Parameters
        ----------
        dataloader :
            dataloader
        classnum :
            classnum
        target_or_not :
            target_or_not
        clip_max :
            maximum pixel value
        clip_min :
            minimum pixel value
        epsilon :
            perturb constraint    
        population :
            population
        max_iterations :
            maximum number of iterations
        learning_rate :
            learning rate
        sigma :
            sigma
        """

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
    succImages = 0
    faillist = []
    successlist = []
    printlist = []

    for i, (inputs, targets) in enumerate(loader):

        success = False
        print('attack picture No. ' + str(i))

        c = inputs.size(1)  # chanel
        l = inputs.size(2)  # length
        w = inputs.size(3)  # width

        mu = arctanh((inputs * 2) - 1)
        #mu = torch.from_numpy(np.random.randn(1, c, l, w) * 0.001).float()  # random initialize mean
        predict = model.forward(inputs)

        ## skip wrongly classified samples
        if  predict.argmax(dim = 1, keepdim = True) != targets:
            print('skip the wrong example ', i)
            continue
        totalImages += 1

        ## finding most possible mean
        for runstep in range(max_iterations):

            # sample points from normal distribution
            eps = torch.from_numpy(np.random.randn(population, c, l, w)).float()
            z = mu.repeat(population, 1, 1, 1) + sigma * eps

            # calculate g_z
            g_z = np.tanh(z) * 1 / 2 + 1 / 2

            # testing whether exists successful attack every 10 iterations.
            if runstep % 10 == 0:

                realdist = g_z - inputs

                realclipdist = np.clip(realdist, -epsilon, epsilon).float()
                realclipinput = realclipdist + inputs

                info = 'inputs.shape__' + str(inputs.shape)
                logging.debug(info)

                predict = model.forward(realclipinput)

                #pending attack
                if (target_or_not == False):

                    if sum(predict.argmax(dim = 1, keepdim = True)[0] != targets) > 0 and (np.abs(realclipdist).max() <= epsilon):
                        succImages += 1
                        success = True
                        print('succeed attack Images: '+str(succImages)+'     totalImages: '+str(totalImages))
                        print('steps: '+ str(runstep))
                        successlist.append(i)
                        printlist.append(runstep)
                        break

            # calculate distance
            dist = g_z - inputs
            clipdist = np.clip(dist, -epsilon, epsilon)
            proj_g_z = inputs + clipdist
            proj_g_z = proj_g_z.float()
            outputs = model.forward(proj_g_z)

            # get cw loss on sampled images
            target_onehot = np.zeros((1,classnum))
            target_onehot[0][targets]=1.
            real = (target_onehot * outputs.detach().numpy()).sum(1)
            other = ((1. - target_onehot) * outputs.detach().numpy() - target_onehot * 10000.).max(1)
            loss1 = np.clip(real - other, a_min= 0, a_max= 1e10)
            Reward = 0.5 * loss1

            # update mean by nes
            A = ((Reward - np.mean(Reward)) / (np.std(Reward)+1e-7))
            A = np.array(A, dtype= np.float32)

            mu = mu - torch.from_numpy((learning_rate/(population*sigma)) *
                                               ((np.dot(eps.reshape(population,-1).T, A)).reshape(1, 1, 28, 28)))

        if not success:
            faillist.append(i)
            print('failed:',faillist.__len__())
            print('....................................')
        else:
            #print('succeed:',successlist.__len__())
            print('....................................')


