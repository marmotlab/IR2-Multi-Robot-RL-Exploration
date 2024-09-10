##############################################################################
# Name: test_driver.py
# [Inference] Driver of training program, maintain & update the global network.
###############################################################################

from test_parameter import *
import ray
import numpy as np
import os
import torch
import csv
import pandas as pd
from model import PolicyNet
from test_multi_robot_worker import TestWorker
from datetime import datetime

def run_test():

    # Create .csv file for data collection
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_file_name = "data_{}.csv".format(current_datetime)
    csv_file_path = os.path.join(log_path, csv_file_name)

    # Create CSV file
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w') as csv_file:
            fieldnames = ['eps', 'num_robots', 'max_dist', 'steps', 'explored', 'success', 'connectivity']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()


    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    if device == 'cuda':
        checkpoint = torch.load(MODEL_PATH)
    else:
        checkpoint = torch.load(MODEL_PATH, map_location = torch.device('cpu'))

    global_network.load_state_dict(checkpoint['policy_model'])

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()
    curr_test = 0

    dist_history = []

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        eps_skipped = []
        while len(dist_history) < NUM_TEST:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                success, metrics, info = job
                if success:
                    dist_history.append(metrics['travel_dist'])

                    # Populate CSV file
                    with open(csv_file_path, mode='a') as csv_file:
                        fieldnames = ['eps', 'num_robots', 'max_dist', 'steps', 'explored', 'success', 'connectivity']
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writerow({'eps': info['episode_number'], \
                                        'num_robots': info['n_agent'], \
                                        'max_dist': metrics['travel_dist'], \
                                        'steps': metrics['travel_steps'], \
                                        'explored': metrics['explored_rate'], \
                                        'success': metrics['success_rate'], \
                                        'connectivity': metrics['connectivity_rate'] })
                else:
                    eps_skipped.append(curr_test)
            if curr_test < (NUM_TEST + len(eps_skipped)):
                job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                curr_test += 1

        # Sort CSV file by episode number
        df = pd.read_csv(csv_file_path)
        sorted_df = df.sort_values(by='eps')
        sorted_df.to_csv(csv_file_path, index=False)

        print('|#Total test:', NUM_TEST)
        print('|#Average (Max) length:', np.array(dist_history).mean())
        print('|#Length std:', np.array(dist_history).std())
        print('|#Eps skipped:', eps_skipped)


    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
 

@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number):
        """ Execute simulation episode and gather experience tuples & metrics """
        n_agent = np.random.randint(NUM_ROBOTS_MIN, NUM_ROBOTS_MAX+1, 1)[0]     
        worker = TestWorker(self.meta_agent_id, n_agent, self.local_network, episode_number, device=self.device, save_image=SAVE_GIFS, greedy=True)
        success = worker.work(episode_number)

        perf_metrics = worker.perf_metrics
        return success, perf_metrics, n_agent

    def job(self, weights, episode_number):
        """ Executes simulation episode """
        print(GREEN, "starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id), NC)
        
        # Set the local weights to the global weight values from the master network
        self.set_weights(weights)

        success, metrics, n_agent = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
            "n_agent": n_agent
        }

        return success, metrics, info


if __name__ == '__main__':
    ray.init()
    print("Welcome to IR2-MARL Exploration Inference Sim!")
    for i in range(NUM_RUN):
        run_test()