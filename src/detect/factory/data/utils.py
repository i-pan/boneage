def cudaify(batch, labels):
    if type(labels) == dict:
        return batch.cuda(), {k:v.cuda() for k,v in labels.items()}
    else:
        return batch.cuda(), labels.cuda()