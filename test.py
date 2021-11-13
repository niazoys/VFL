import random
bs=5
seen_sample=[]

def generate_batch_ids(limit,n_samples,batch_size):
    ids=[]
    counter=0
    r = random.sample(range(limit), n_samples)
    for e in r:
        if e not in seen_sample:
            seen_sample.append(e)
            ids.append(e)
            counter=counter+1
            if counter==batch_size:
                return ids

for _ in range(5):
    print(generate_batch_ids())