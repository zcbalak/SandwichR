import random
def sample_by_label(data, target_count):
    sampled_data = []
    label_samples = {}
    
    for item in data:
        label = item['metadata']['label']
        if label not in label_samples:
            label_samples[label] = []
        label_samples[label].append(item)
    
    for label, samples in label_samples.items():
        if len(samples) > target_count:
            samples = random.sample(samples, target_count)
        sampled_data.extend(samples)
    
    return sampled_data
 
