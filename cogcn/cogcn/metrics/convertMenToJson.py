
import json
import pickle
from pathlib import Path

with open('C:\\python\\cogcn\\cogcn\\datasets_runtime1\\acmeair\\cogcn_output\\embeddings_membership.pkl','rb') as f:
    clustering = pickle.load(f).tolist()
    # print(clustering)
    
with open("C:\python\cogcn\cogcn\datasets_runtime1\\acmeair\cogcn_output\mapping.json") as mapping_file:
    mapping = json.load(mapping_file)
    # print(mapping)


vertical_cluster_assignment = {class_name: 
                    clustering[int(id_)] for id_, class_name in mapping.items()}

with open(Path("C:\python\cogcn\cogcn\datasets_runtime1\\acmeair\cogcn_output").joinpath('vertical_cluster_assignment__{}.json'.format(6)), 'w') as cluster_assgn_file:
                json.dump(vertical_cluster_assignment,
                          cluster_assgn_file, indent=4)