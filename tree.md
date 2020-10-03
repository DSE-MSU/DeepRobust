.
├── LICENSE
├── README.md
├── adversary_examples
│   ├── cifar_advexample_orig.png
│   └── cifar_advexample_pgd.png
├── deeprobust
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-36.pyc
│   │   └── __init__.cpython-37.pyc
│   ├── graph
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── __init__.pyc
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-36.pyc
│   │   │   ├── __init__.cpython-37.pyc
│   │   │   ├── black_box.cpython-36.pyc
│   │   │   ├── black_box.cpython-37.pyc
│   │   │   ├── utils.cpython-36.pyc
│   │   │   └── utils.cpython-37.pyc
│   │   ├── black_box.py
│   │   ├── data
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-36.pyc
│   │   │   │   ├── __init__.cpython-37.pyc
│   │   │   │   ├── attacked_data.cpython-36.pyc
│   │   │   │   ├── attacked_data.cpython-37.pyc
│   │   │   │   ├── dataset.cpython-36.pyc
│   │   │   │   └── dataset.cpython-37.pyc
│   │   │   ├── attacked_data.py
│   │   │   └── dataset.py
│   │   ├── defense
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-36.pyc
│   │   │   │   ├── __init__.cpython-37.pyc
│   │   │   │   ├── gcn.cpython-36.pyc
│   │   │   │   ├── gcn.cpython-37.pyc
│   │   │   │   ├── gcn_preprocess.cpython-36.pyc
│   │   │   │   ├── gcn_preprocess.cpython-37.pyc
│   │   │   │   ├── r_gcn.cpython-36.pyc
│   │   │   │   └── r_gcn.cpython-37.pyc
│   │   │   ├── adv_training.py
│   │   │   ├── gcn.py
│   │   │   ├── gcn_preprocess.py
│   │   │   ├── r_gcn.py
│   │   │   └── r_gcn.py.backup
│   │   ├── examples
│   │   │   ├── __pycache__
│   │   │   │   ├── test_adv_train_evasion.cpython-36.pyc
│   │   │   │   ├── test_adv_train_evasion.cpython-37.pyc
│   │   │   │   ├── test_adv_train_poisoning.cpython-36.pyc
│   │   │   │   ├── test_adv_train_poisoning.cpython-37.pyc
│   │   │   │   ├── test_adv_training.cpython-36.pyc
│   │   │   │   ├── test_dice.cpython-36.pyc
│   │   │   │   ├── test_dice.cpython-37.pyc
│   │   │   │   ├── test_fgsm.cpython-36.pyc
│   │   │   │   ├── test_fgsm.cpython-37.pyc
│   │   │   │   ├── test_gcn.cpython-36.pyc
│   │   │   │   ├── test_gcn.cpython-37.pyc
│   │   │   │   ├── test_gcn_jaccard.cpython-36.pyc
│   │   │   │   ├── test_gcn_jaccard.cpython-37.pyc
│   │   │   │   ├── test_gcn_svd.cpython-36.pyc
│   │   │   │   ├── test_gcn_svd.cpython-37.pyc
│   │   │   │   ├── test_mettack.cpython-36.pyc
│   │   │   │   ├── test_mettack.cpython-37.pyc
│   │   │   │   ├── test_nettack.cpython-36.pyc
│   │   │   │   ├── test_nettack.cpython-37.pyc
│   │   │   │   ├── test_nipa.cpython-36.pyc
│   │   │   │   ├── test_nipa.cpython-37.pyc
│   │   │   │   ├── test_random.cpython-36.pyc
│   │   │   │   ├── test_random.cpython-37.pyc
│   │   │   │   ├── test_rgcn.cpython-36.pyc
│   │   │   │   ├── test_rgcn.cpython-37.pyc
│   │   │   │   ├── test_rl_s2v.cpython-36.pyc
│   │   │   │   ├── test_rl_s2v.cpython-37.pyc
│   │   │   │   ├── test_rnd.cpython-36.pyc
│   │   │   │   ├── test_rnd.cpython-37.pyc
│   │   │   │   └── test_topology_attack.cpython-36.pyc
│   │   │   ├── test_adv_train_evasion.py
│   │   │   ├── test_adv_train_poisoning.py
│   │   │   ├── test_dice.py
│   │   │   ├── test_fgsm.py
│   │   │   ├── test_gcn.py
│   │   │   ├── test_gcn_jaccard.py
│   │   │   ├── test_gcn_svd.py
│   │   │   ├── test_mettack.py
│   │   │   ├── test_nettack.py
│   │   │   ├── test_nipa.py
│   │   │   ├── test_random.py
│   │   │   ├── test_rgcn.py
│   │   │   ├── test_rl_s2v.py
│   │   │   ├── test_rnd.py
│   │   │   └── test_topology_attack.py
│   │   ├── global_attack
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-36.pyc
│   │   │   │   ├── __init__.cpython-37.pyc
│   │   │   │   ├── base_attack.cpython-36.pyc
│   │   │   │   ├── base_attack.cpython-37.pyc
│   │   │   │   ├── dice.cpython-36.pyc
│   │   │   │   ├── dice.cpython-37.pyc
│   │   │   │   ├── mettack.cpython-36.pyc
│   │   │   │   ├── mettack.cpython-37.pyc
│   │   │   │   ├── nipa.cpython-36.pyc
│   │   │   │   ├── random.cpython-36.pyc
│   │   │   │   ├── random.cpython-37.pyc
│   │   │   │   └── topology_attack.cpython-36.pyc
│   │   │   ├── base_attack.py
│   │   │   ├── dice.py
│   │   │   ├── mettack.py
│   │   │   ├── nipa.py
│   │   │   ├── random.py
│   │   │   └── topology_attack.py
│   │   ├── requirements.txt
│   │   ├── rl
│   │   │   ├── __pycache__
│   │   │   │   ├── agent.cpython-36.pyc
│   │   │   │   ├── agent.cpython-37.pyc
│   │   │   │   ├── cmd_args.cpython-36.pyc
│   │   │   │   ├── cmd_args.cpython-37.pyc
│   │   │   │   ├── env.cpython-36.pyc
│   │   │   │   ├── env.cpython-37.pyc
│   │   │   │   ├── nipa.cpython-36.pyc
│   │   │   │   ├── nipa.cpython-37.pyc
│   │   │   │   ├── nipa_args.cpython-37.pyc
│   │   │   │   ├── nipa_config.cpython-36.pyc
│   │   │   │   ├── nipa_config.cpython-37.pyc
│   │   │   │   ├── nipa_env.cpython-36.pyc
│   │   │   │   ├── nipa_env.cpython-37.pyc
│   │   │   │   ├── nipa_nstep_replay_mem.cpython-36.pyc
│   │   │   │   ├── nipa_q_net_node.cpython-36.pyc
│   │   │   │   ├── nipa_q_net_node.cpython-37.pyc
│   │   │   │   ├── nstep_replay_mem.cpython-36.pyc
│   │   │   │   ├── nstep_replay_mem.cpython-37.pyc
│   │   │   │   ├── q_net_node.cpython-36.pyc
│   │   │   │   ├── q_net_node.cpython-37.pyc
│   │   │   │   ├── rl_s2v_config.cpython-36.pyc
│   │   │   │   └── rl_s2v_config.cpython-37.pyc
│   │   │   ├── env.py
│   │   │   ├── nipa.py
│   │   │   ├── nipa_config.py
│   │   │   ├── nipa_env.py
│   │   │   ├── nipa_nstep_replay_mem.py
│   │   │   ├── nipa_q_net_node.py
│   │   │   ├── nstep_replay_mem.py
│   │   │   ├── q_net_node.py
│   │   │   ├── rl_s2v.py
│   │   │   ├── rl_s2v_config.py
│   │   │   └── rl_s2v_env.py
│   │   ├── targeted_attack
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-36.pyc
│   │   │   │   ├── __init__.cpython-37.pyc
│   │   │   │   ├── base_attack.cpython-36.pyc
│   │   │   │   ├── base_attack.cpython-37.pyc
│   │   │   │   ├── evaluation.cpython-36.pyc
│   │   │   │   ├── fgsm.cpython-36.pyc
│   │   │   │   ├── fgsm.cpython-37.pyc
│   │   │   │   ├── nettack.cpython-36.pyc
│   │   │   │   ├── nettack.cpython-37.pyc
│   │   │   │   ├── rl_s2v.cpython-36.pyc
│   │   │   │   ├── rnd.cpython-36.pyc
│   │   │   │   └── rnd.cpython-37.pyc
│   │   │   ├── base_attack.py
│   │   │   ├── evaluation.py
│   │   │   ├── fgsm.py
│   │   │   ├── nettack.py
│   │   │   ├── rl_s2v.py
│   │   │   └── rnd.py
│   │   └── utils.py
│   └── image
│       ├── README.md
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── CNNmodel.cpython-36.pyc
│       │   ├── PGD.cpython-36.pyc
│       │   ├── __init__.cpython-36.pyc
│       │   ├── __init__.cpython-37.pyc
│       │   ├── attack.cpython-36.pyc
│       │   ├── deepfool.cpython-36.pyc
│       │   ├── evaluation_attack.cpython-36.pyc
│       │   ├── fgsm.cpython-36.pyc
│       │   ├── lbfgs.cpython-36.pyc
│       │   ├── mnist_train.cpython-36.pyc
│       │   ├── optimizer.cpython-36.pyc
│       │   └── utils.cpython-36.pyc
│       ├── adversary_examples
│       │   ├── advexample.png
│       │   ├── advexample_fgsm.png
│       │   ├── advexample_pgd.png
│       │   ├── cifar_advexample_orig.png
│       │   ├── cifar_advexample_pgd.png
│       │   ├── deepfool_diff.png
│       │   ├── deepfool_orig.png
│       │   ├── deepfool_pert.png
│       │   ├── deepfool_result.png
│       │   ├── imageexample.png
│       │   ├── test.jpg
│       │   ├── test.png
│       │   ├── test1.jpg
│       │   ├── test_im1.jpg
│       │   └── test_im2.jpg
│       ├── attack
│       │   ├── BPDA.py
│       │   ├── Nattack.py
│       │   ├── Universal.py
│       │   ├── YOPOpgd.py
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── YOPOpgd.cpython-36.pyc
│       │   │   ├── __init__.cpython-36.pyc
│       │   │   ├── __init__.cpython-37.pyc
│       │   │   ├── attack.cpython-36.pyc
│       │   │   ├── base_attack.cpython-36.pyc
│       │   │   ├── base_attack.cpython-37.pyc
│       │   │   ├── cw.cpython-36.pyc
│       │   │   ├── deepfool.cpython-36.pyc
│       │   │   ├── fgsm.cpython-36.pyc
│       │   │   ├── lbfgs.cpython-36.pyc
│       │   │   └── pgd.cpython-36.pyc
│       │   ├── base_attack.py
│       │   ├── cw.py
│       │   ├── deepfool.py
│       │   ├── fgsm.py
│       │   ├── l2_attack.py
│       │   ├── lbfgs.py
│       │   ├── onepixel.py
│       │   └── pgd.py
│       ├── config.py
│       ├── data
│       ├── defense
│       │   ├── LIDclassifier.py
│       │   ├── TherEncoding.py
│       │   ├── YOPO.py
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── TherEncoding.cpython-36.pyc
│       │   │   ├── YOPO.cpython-36.pyc
│       │   │   ├── __init__.cpython-36.pyc
│       │   │   ├── base_defense.cpython-36.pyc
│       │   │   ├── fgsmtraining.cpython-36.pyc
│       │   │   ├── pgdtraining.cpython-36.pyc
│       │   │   ├── trade.cpython-36.pyc
│       │   │   └── trades.cpython-36.pyc
│       │   ├── advexample_pgd.png
│       │   ├── base_defense.py
│       │   ├── fast.py
│       │   ├── fgsmtraining.py
│       │   ├── pgdtraining.py
│       │   ├── test_PGD_defense.py
│       │   ├── trade.py
│       │   └── trades.py
│       ├── evaluation_attack.py
│       ├── netmodels
│       │   ├── CNN.py
│       │   ├── CNN_multilayer.py
│       │   ├── YOPOCNN.py
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── CNN.cpython-36.pyc
│       │   │   ├── CNNmodel.cpython-36.pyc
│       │   │   ├── YOPOCNN.cpython-36.pyc
│       │   │   ├── __init__.cpython-36.pyc
│       │   │   ├── resnet.cpython-36.pyc
│       │   │   └── train_model.cpython-36.pyc
│       │   ├── resnet.py
│       │   ├── train_model.py
│       │   └── train_resnet.py
│       ├── optimizer.py
│       ├── synset_words.txt
│       └── utils.py
├── examples
│   ├── graph
│   │   ├── test_adv_train_evasion.py
│   │   ├── test_adv_train_poisoning.py
│   │   ├── test_dice.py
│   │   ├── test_fgsm.py
│   │   ├── test_gcn.py
│   │   ├── test_gcn_jaccard.py
│   │   ├── test_gcn_svd.py
│   │   ├── test_mettack.py
│   │   ├── test_nettack.py
│   │   ├── test_nipa.py
│   │   ├── test_random.py
│   │   ├── test_rgcn.py
│   │   ├── test_rl_s2v.py
│   │   └── test_rnd.py
│   └── image
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-36.pyc
│       │   └── test_cw.cpython-36.pyc
│       ├── test1.py
│       ├── test_PGD.py
│       ├── test_cw.py
│       ├── test_deepfool.py
│       ├── test_fgsm.py
│       ├── test_lbfgs.py
│       ├── test_nattack.py
│       ├── test_onepixel.py
│       ├── test_pgdtraining.py
│       ├── test_trade.py
│       ├── test_train.py
│       └── testprint_mnist.py
├── get-pip.py
├── requirements.txt
├── setup.py
├── tree.md
└── tutorials
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-36.pyc
    │   └── test_cw.cpython-36.pyc
    ├── test1.py
    ├── test_PGD.py
    ├── test_cw.py
    ├── test_deepfool.py
    ├── test_fgsm.py
    ├── test_lbfgs.py
    ├── test_nattack.py
    ├── test_onepixel.py
    ├── test_pgdtraining.py
    ├── test_trade.py
    ├── test_train.py
    └── testprint_mnist.py

33 directories, 297 files
