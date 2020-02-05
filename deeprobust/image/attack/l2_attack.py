import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CarliniL2:
    def __init__(self, model, device): 
        self.model = model
        self.device = device

    def parse_params(self, gan, confidence=0, targeted=False, learning_rate=1e-1,
                 binary_search_steps=5, max_iterations=10000, abort_early=False, initial_const=1,
                 clip_min=0, clip_max=1):
        
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.gan = gan
        self.learning_rate = learning_rate
        self.repeat = binary_search_steps >= 10

    def get_or_guess_labels(self, x, y=None):
        """
        Get the label to use in generating an adversarial example for x.
        The kwargs are fed directly from the kwargs of the attack.
        If 'y' is in kwargs, use that as the label.
        Otherwise, use the model's prediction as the label.
        """
        if y is not None:
            labels = y
        else:
            preds = F.softmax(self.model(x))
            preds_max = torch.max(preds, 1, keepdim=True)[0]
            original_predictions = (preds == preds_max)
            labels = original_predictions
            del preds
        return labels.float()

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def to_one_hot(self, x):
        one_hot = torch.FloatTensor(x.shape[0], 10).to(x.get_device())
        one_hot.zero_()
        x = x.unsqueeze(1)
        one_hot = one_hot.scatter_(1, x, 1)
        return one_hot

    def generate(self, imgs, y, start):

        batch_size = imgs.shape[0]
        labs = self.get_or_guess_labels(imgs, y)

        def compare(x, y):
            if self.TARGETED is None: return True

            if sum(x.shape) != 0:
                x = x.clone()
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = torch.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        # set the lower and upper bounds accordingly
        lower_bound = torch.zeros(batch_size).to(self.device)
        CONST = torch.ones(batch_size).to(self.device) * self.initial_const
        upper_bound = (torch.ones(batch_size) * 1e10).to(self.device)

        # the best l2, score, and image attack
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = self.gan(start)

        # check if the input label is one-hot, if not, then change it into one-hot vector
        if len(labs.shape) == 1:
            tlabs = self.to_one_hot(labs.long())
        else:
            tlabs = labs

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            modifier = nn.Parameter(start)
            optimizer = torch.optim.Adam([modifier, ], lr=self.learning_rate)

            bestl2 = [1e10] * batch_size
            bestscore = -1 * torch.ones(batch_size, dtype=torch.float32).to(self.device)

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound
            prev = 1e6

            for i in range(self.MAX_ITERATIONS):
                optimizer.zero_grad()
                nimgs = self.gan(modifier.to(self.device))

                # distance to the input data
                l2dist = torch.sum(torch.sum(torch.sum((nimgs - imgs) ** 2, 1), 1), 1)
                loss2 = torch.sum(l2dist)

                # prediction BEFORE-SOFTMAX of the model
                scores = self.model(nimgs)

                # compute the probability of the label class versus the maximum other
                other = torch.max(((1 - tlabs) * scores - tlabs * 10000), 1)[0]
                real = torch.sum(tlabs * scores, 1)

                if self.TARGETED:
                    # if targeted, optimize for making the other class most likely
                    loss1 = torch.max(torch.zeros_like(other), other - real + self.CONFIDENCE)
                else:
                    # if untargeted, optimize for making this class least likely.
                    loss1 = torch.max(torch.zeros_like(other), real - other + self.CONFIDENCE)

                # sum up the losses
                loss1 = torch.sum(CONST * loss1)
                loss = loss1 + loss2

                # update the modifier
                loss.backward()
                optimizer.step()

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and i % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    if loss > prev * .9999:
                        # print('Stop early')
                        break
                    prev = loss

                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2dist, scores, nimgs)):
                    lab = torch.argmax(tlabs[e])

                    if l2 < bestl2[e] and compare(sc, lab):
                        bestl2[e] = l2
                        bestscore[e] = torch.argmax(sc)

                    if l2 < o_bestl2[e] and compare(sc, lab):
                        o_bestl2[e] = l2
                        o_bestscore[e] = torch.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], torch.argmax(tlabs[e]).float()) and \
                        bestscore[e] != -1:
                    # success, divide CONST by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack