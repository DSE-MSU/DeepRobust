from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import DeepWalk
from deeprobust.graph.global_attack import NodeEmbeddingAttack
from deeprobust.graph.global_attack import OtherNodeEmbeddingAttack
import itertools

dataset_str = 'cora_ml'
data = Dataset(root='/tmp/', name=dataset_str, seed=15)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

comb = itertools.product(["random", "degree", "eigencentrality"], ["remove", "add"])
for type, attack_type in comb:
    model = OtherNodeEmbeddingAttack(type=type)
    print(model.type, attack_type)
    try:
        model.attack(adj, attack_type=attack_type, n_candidates=10000)
        modified_adj = model.modified_adj
        defender = DeepWalk()
        defender.fit(modified_adj)
        defender.evaluate_node_classification(labels, idx_train, idx_test)
    except KeyError:
        print('eigencentrality only supports removing edges')

model = NodeEmbeddingAttack()
model.attack(adj, attack_type="remove")
model.attack(adj, attack_type="remove", min_span_tree=True)
modified_adj = model.modified_adj
model.attack(adj, attack_type="add", n_candidates=10000)
model.attack(adj, attack_type="add_by_remove", n_candidates=10000)
