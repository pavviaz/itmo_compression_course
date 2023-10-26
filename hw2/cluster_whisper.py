from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load
from itertools import islice
from tqdm import tqdm
from time import time
from collections import defaultdict
import itertools


HF_CACHE = "/raid/nanosemantics/vyaznikov/itmo_hw/itmo_compression_course/hw3"


processor = WhisperProcessor.from_pretrained("openai/whisper-base", cache_dir=HF_CACHE)
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-base", cache_dir=HF_CACHE
).to("cpu")


from sklearn.cluster import KMeans


def weight_clustering(model):
    model.to("cpu")
    with torch.no_grad():
        count = 0
        for name, params in model.named_parameters():
            param_shape = list(params.size())
            weights = params.reshape(-1, 1)
            kmeans = KMeans(n_clusters=5, random_state=0).fit(weights)
            cluster_centers = torch.from_numpy(kmeans.cluster_centers_)
            print("Processing for layer ", count)
            count += 1
            # print(type(cluster_centers))
            cluster_list = []
            for i in range(0, len(kmeans.labels_)):
                if kmeans.labels_[i] == 0:
                    cluster_list.append(cluster_centers[0].view(1))
                    # print(cluster_list[i].data_ptr() == cluster_centers[0].data_ptr())
                elif kmeans.labels_[i] == 1:
                    cluster_list.append(cluster_centers[1].view(1))
                    # print(cluster_list[i].data_ptr() == cluster_centers[1].data_ptr())
                elif kmeans.labels_[i] == 2:
                    cluster_list.append(cluster_centers[2].view(1))
                    # print(cluster_list[i].data_ptr() == cluster_centers[2].data_ptr())
                elif kmeans.labels_[i] == 3:
                    cluster_list.append(cluster_centers[3].view(1))
                    # print(cluster_list[i].data_ptr() == cluster_centers[3].data_ptr())
                elif kmeans.labels_[i] == 4:
                    cluster_list.append(cluster_centers[4].view(1))
                    # print(cluster_list[i].data_ptr() == cluster_centers[4].data_ptr())

            reshape_size_tuple = tuple(param_shape)
            cluster_list = torch.tensor(cluster_list)
            cluster_list = cluster_list.reshape(reshape_size_tuple)
            params.data = cluster_list.data
            print(params.data_ptr() == cluster_list.data_ptr())

    return model


cl_model = weight_clustering(model)
