Node Embedding Attack and Defense 
=======================
In this section, we introduce the node embedding attack algorithms and 
corresponding victim models provided in DeepRobust. 

.. contents::
    :local: 


Node Embedding Attack
-----------------------
Node embedding attack aims to fool node embedding models produce bad-quality embeddings. 
Specifically, DeepRobust provides the following node attack algorithms:

- :class:`deeprobust.graph.global_attack.NodeEmbeddingAttack`
- :class:`deeprobust.graph.global_attack.OtherNodeEmbeddingAttack`

They only take the adjacency matrix as input and the adjacency 
matrix is in the format of :obj:`scipy.sparse.csr_matrix`. You can specify the attack_type
to either add edges or remove edges. Let's take a look at an example:

.. code-block:: python

    from deeprobust.graph.data import Dataset
    from deeprobust.graph.global_attack import NodeEmbeddingAttack
    data = Dataset(root='/tmp/', name='cora_ml', seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    model = NodeEmbeddingAttack()
    model.attack(adj, attack_type="remove")
    modified_adj = model.modified_adj
    model.attack(adj, attack_type="remove", min_span_tree=True)
    modified_adj = model.modified_adj
    model.attack(adj, attack_type="add", n_candidates=10000)
    modified_adj = model.modified_adj
    model.attack(adj, attack_type="add_by_remove", n_candidates=10000)
    modified_adj = model.modified_adj

The :obj:`OtherNodeEmbeddingAttack` contains the baseline methods reported in the paper 
Adversarial Attacks on Node Embeddings via Graph Poisoning. Aleksandar Bojchevski and 
Stephan Günnemann, ICML 2019. We can specify the type (chosen from 
`["degree", "eigencentrality", "random"]`) to generate corresponding attacks. 

.. code-block:: python

    from deeprobust.graph.data import Dataset
    from deeprobust.graph.global_attack import OtherNodeEmbeddingAttack
    data = Dataset(root='/tmp/', name='cora_ml', seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    model = OtherNodeEmbeddingAttack(type='degree')
    model.attack(adj, attack_type="remove")
    modified_adj = model.modified_adj
    #
    model = OtherNodeEmbeddingAttack(type='eigencentrality')
    model.attack(adj, attack_type="remove")
    modified_adj = model.modified_adj
    #
    model = OtherNodeEmbeddingAttack(type='random')
    model.attack(adj, attack_type="add", n_candidates=10000)
    modified_adj = model.modified_adj

Node Embedding Victim Models
-----------------------
DeepRobust provides two node embedding victim models, DeepWalk and Node2Vec: 

- :class:`deeprobust.graph.defense.DeepWalk`
- :class:`deeprobust.graph.defense.Node2Vec`

There are three major functions in the two classes: :obj:`fit()`, :obj:`evaluate_node_classification()` 
and :obj:`evaluate_link_prediction`. The function :obj:`fit()` will train the node embdding models 
and store the embedding in :obj:`self.embedding`. For example,

.. code-block:: python

    from deeprobust.graph.data import Dataset
    from deeprobust.graph.defense import DeepWalk
    from deeprobust.graph.global_attack import NodeEmbeddingAttack
    import numpy as np

    dataset_str = 'cora_ml'
    data = Dataset(root='/tmp/', name=dataset_str, seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    print("Test DeepWalk on clean graph")
    model = DeepWalk(type="skipgram")
    model.fit(adj)
    print(model.embedding)

After we trained the model, we can then test its performance on node classification and link prediction:

.. code-block:: python

    print("Test DeepWalk on node classification...")
    # model.evaluate_node_classification(labels, idx_train, idx_test, lr_params={"max_iter": 1000})
    model.evaluate_node_classification(labels, idx_train, idx_test)
    print("Test DeepWalk on link prediciton...")
    model.evaluate_link_prediction(adj, np.array(adj.nonzero()).T)

We can then test its performance on the attacked graph: 

.. code-block:: python

    # set up the attack model
    attacker = NodeEmbeddingAttack()
    attacker.attack(adj, attack_type="remove", n_perturbations=1000)
    modified_adj = attacker.modified_adj
    print("Test DeepWalk on attacked graph")
    model.fit(modified_adj)
    model.evaluate_node_classification(labels, idx_train, idx_test)

