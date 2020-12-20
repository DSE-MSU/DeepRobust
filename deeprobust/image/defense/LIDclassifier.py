"""
This is an implementation of LID detector.
Currently this implementation is under testing.

References
----------
.. [1] Ma, Xingjun, Bo Li, Yisen Wang, Sarah M. Erfani, Sudanthi Wijewickrema, Grant Schoenebeck, Dawn Song, Michael E. Houle, and James Bailey. "Characterizing adversarial subspaces using local intrinsic dimensionality." arXiv preprint arXiv:1801.02613 (2018).
.. [2] Original code:t https://github.com/xingjunm/lid_adversarial_subspace_detection
Copyright (c) 2018 Xingjun Ma
"""

from deeprobust.image.netmodels.CNN_multilayer import Net

def train(self, device, train_loader, optimizer, epoch):
    """train process.

    Parameters
    ----------
    device :
        device(option:'cpu', 'cuda')
    train_loader :
        train data loader
    optimizer :
        optimizer
    epoch :
        epoch
    """
    self.model.train()
    correct = 0
    bs = train_loader.batch_size

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)

        data_adv, output = self.adv_data(data, target, ep = self.epsilon, num_steps = self.num_steps)

        loss = self.calculate_loss(output, target)

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim = 1, keepdim = True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        #print every 10
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy:{:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), 100 * correct/(10*bs)))
        correct = 0

def get_lid(model, X_test, X_test_noisy, X_test_adv, k, batch_size):
    """get_lid.

    Parameters
    ----------
    model :
        model
    X_test :
        clean data
    X_test_noisy :
        noisy data
    X_test_adv :
        adversarial data
    k :
        k
    batch_size :
        batch_size
    """
    funcs = [K.function([model.layers[0].input, K.learning_phase()], [out])
                 for out in get_layer_wise_activations(model, dataset)]

    lid_dim = len(funcs)
    print("Number of layers to estimate: ", lid_dim)

    def estimate(i_batch):

        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_adv = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_noisy = np.zeros(shape=(n_feed, lid_dim))

        for i, func in enumerate(funcs):
            X_act = func([X[start:end], 0])[0]
            X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_act: ", X_act.shape)

            X_adv_act = func([X_adv[start:end], 0])[0]
            X_adv_act = np.asarray(X_adv_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_adv_act: ", X_adv_act.shape)

            X_noisy_act = func([X_noisy[start:end], 0])[0]
            X_noisy_act = np.asarray(X_noisy_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_noisy_act: ", X_noisy_act.shape)

            # random clean samples
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch[:, i] = mle_batch(X_act, X_act, k=k)
            # print("lid_batch: ", lid_batch.shape)
            lid_batch_adv[:, i] = mle_batch(X_act, X_adv_act, k=k)
            # print("lid_batch_adv: ", lid_batch_adv.shape)
            lid_batch_noisy[:, i] = mle_batch(X_act, X_noisy_act, k=k)
            # print("lid_batch_noisy: ", lid_batch_noisy.shape)

        return lid_batch, lid_batch_noisy, lid_batch_adv

    lids = []
    lids_adv = []
    lids_noisy = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))

    for i_batch in tqdm(range(n_batches)):

        lid_batch, lid_batch_noisy, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)
        lids_noisy.extend(lid_batch_noisy)
        # print("lids: ", lids.shape)
        # print("lids_adv: ", lids_noisy.shape)
        # print("lids_noisy: ", lids_noisy.shape)

    lids_normal = np.asarray(lids, dtype=np.float32)
    lids_noisy = np.asarray(lids_noisy, dtype=np.float32)
    lids_adv = np.asarray(lids_adv, dtype=np.float32)

    lids_pos = lids_adv
    lids_neg = np.concatenate((lids_normal, lids_noisy))
    artifacts, labels = merge_and_generate_labels(lids_pos, lids_neg)

    return artifacts, labels

if __name__ == "__main__":

    batch_size = 100
    k_nearest = 20

    #get LID characters
    characters, labels = get_lid(model, X_test, X_test_noisy, X_test_adv, k_nearest, batch_size)
    data = np.concatenate((characters, labels), axis = 1)

