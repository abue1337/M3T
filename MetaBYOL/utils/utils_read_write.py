import json
from MetaBYOL.utils import utils_plots

def write_loss_acc_to_file(run_paths, loss, acc):
    json_string1 = json.dumps(str(loss))
    json_string2 = json.dumps(str(acc))
    with open(run_paths['path_graphs_eval']+'loss.json', 'w') as f:
        json.dump(json_string1, f)
    with open(run_paths['path_graphs_eval']+'acc.json', 'w') as f:
        json.dump(json_string2, f)


def read_loss_acc_from_file(path):
    with open(path+'\\evalacc.json') as json_file:
        a = json.loads(json.load(json_file))
    with open(path+'\\evalloss.json') as json_file:
        b = json.loads(json.load(json_file))
    return json.loads(a), json.loads(b)



