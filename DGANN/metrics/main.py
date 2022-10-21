'''
Created on Mar 27, 2021

@author: kaliaanup
'''
import pathlib
import metrics_util
import json
import os
import sys
from pathlib import Path

# Add project source to path
root = Path(os.path.abspath(os.path.join(
    os.getcwd().split('src')[0], 'src')))

if __name__ == "__main__": 
    
    # PROJECT_DIR = root.joinpath("datasets_runtime")
    PROJECT_DIR = "C:\\python\\icse-deeply\\datasets_runtime\\"
    
    #app name
    app = "acmeair"
    benchmark_type = "deeply"
    ROOT = 'Root'
    
    #from mono2micro output dir get partition file, bcs_per_class, runtime_call_volume
    OUTPUT_DIR = os.path.join(PROJECT_DIR, app, benchmark_type+"_output")
    
    # !! CLIENT DATA !! #
    # partition_sizes = {
    #     'acmeair': range(3,14,2),
    #     'daytrader': range(3,14,2),
    #     'jpetstore': range(3,14,2),
    #     'plants': range(3,14,2),
    # }
    # !! CLIENT DATA !! #

    # bcs_per_class = {}
    # with open(PROJECT_DIR.joinpath(app, "mono2micro_output", "bcs_per_class.json"), 'r') as f:
    #     bcs_per_class = json.load(f)
    
    # runtime_call_volume = {}
    # with open(PROJECT_DIR.joinpath(app, "mono2micro_output", "runtime_call_volume.json"), 'r') as f:
    #     runtime_call_volume = json.load(f)
    
    bcs_per_class = {}
    with open(os.path.join(PROJECT_DIR, app, "mono2micro_output", "bcs_per_class.json"), 'r') as f:
        bcs_per_class = json.load(f)
    
    runtime_call_volume = {}
    with open(os.path.join(PROJECT_DIR, app, "mono2micro_output", "runtime_call_volume.json"), 'r') as f:
        runtime_call_volume = json.load(f)
    
    
    partition = {}
    
    #generate metrics for mono2micro
    if benchmark_type == "mono2micro":
        with open(os.path.join(OUTPUT_DIR, "vertical_cluster_assignment.json"), 'r') as f:
            partition = json.load(f)
    
    
        class_bcs_partition_assignment, partition_class_bcs_assignment = metrics_util.gen_class_assignment(partition, bcs_per_class)
        print(partition_class_bcs_assignment)
        bcs = metrics_util.business_context_purity(partition_class_bcs_assignment)
        icp = metrics_util.inter_call_percentage(ROOT, class_bcs_partition_assignment, runtime_call_volume)
        sm = metrics_util.structural_modularity(partition_class_bcs_assignment,runtime_call_volume)
        #mq = metrics_util.modularity_quality(ROOT, partition, runtime_call_volume)
        mq = metrics_util.modular_quality(ROOT, partition_class_bcs_assignment,runtime_call_volume)
        ifn = metrics_util.interface_number(ROOT, partition_class_bcs_assignment,runtime_call_volume)
        
        print("-------------m2m metrics--------------")
        print(str(bcs)+","+str(icp)+","+str(sm)+","+str(mq)+","+str(ifn))

    #generate metrics for cogcn
    if benchmark_type == "deeply":
        print("Partitions,BCS,ICP,SM,MQ,IFN")
        partition_sizes = [4]
        for k in partition_sizes:
            with open(os.path.join(OUTPUT_DIR, "vertical_cluster_assignment__{}.json".format(k)), 'r') as f:
                partition = json.load(f)
        
        
            class_bcs_partition_assignment, partition_class_bcs_assignment = metrics_util.gen_class_assignment(partition, bcs_per_class)
            print(partition_class_bcs_assignment)
            bcs = metrics_util.business_context_purity(partition_class_bcs_assignment)
            icp = metrics_util.inter_call_percentage(ROOT, class_bcs_partition_assignment, runtime_call_volume)
            sm = metrics_util.structural_modularity(partition_class_bcs_assignment,runtime_call_volume)
            mq = metrics_util.modular_quality(ROOT, partition_class_bcs_assignment,runtime_call_volume)
            ifn = metrics_util.interface_number(ROOT, partition_class_bcs_assignment,runtime_call_volume)
            ned = metrics_util.non_extreme_distribution(partition_class_bcs_assignment)
            
            print("-------------cogcn metrics--------------")
            print(str(k)+","+str(bcs)+","+str(icp)+","+str(sm)+","+str(mq)+","+str(ifn)+","+str(ned))

    
    if benchmark_type == "fosci":
        OUTPUT_DIR = os.path.join(PROJECT_DIR, app, benchmark_type+"_output")
        
        partition_sizes = [3,5,7,9,11,13,15,17,19,21]
        
        for k in partition_sizes:
            with open(os.path.join(OUTPUT_DIR, "daytrader_n_candidate_5_repeat_4.csv"), "r") as f:
                partition_file = f.readlines() 
             
                for line in partition_file:
                    line = line.replace("\n", "")
                    class_name, partition_id = line.split(",")
                    partition[class_name] = partition_id
                    
            class_bcs_partition_assignment, partition_class_bcs_assignment = metrics_util.gen_class_assignment(partition, bcs_per_class)
            bcs = metrics_util.business_context_purity(partition_class_bcs_assignment)
            icp = metrics_util.inter_call_percentage(ROOT, class_bcs_partition_assignment, runtime_call_volume)
            sm = metrics_util.structural_modularity(partition_class_bcs_assignment,runtime_call_volume)
            #mq = metrics_util.modularity_quality(ROOT, partition, runtime_call_volume)
            mq = metrics_util.modular_quality(ROOT, partition_class_bcs_assignment,runtime_call_volume)
            ifn = metrics_util.interface_number(ROOT, partition_class_bcs_assignment,runtime_call_volume)
            
            print("-------------fosci metrics--------------")
            print(str(bcs)+","+str(icp)+","+str(sm)+","+str(mq)+","+str(ifn))
    
    
    
    