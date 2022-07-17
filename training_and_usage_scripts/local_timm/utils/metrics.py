""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def get_sorted_max_over_t(outs_over_t):
    """
    Given predictions over time (list of len=Time and each tensor of dim = Batch x Classes), this sorts predictions by putting the least confident as the first row and most confident as last row
    Note: Returns as a list of tensors 
    """
    outs_over_t = torch.stack(outs_over_t)
    outs_over_t = outs_over_t.transpose(1,0)
    for i in range(outs_over_t.shape[0]):
        outs_over_t[i] = torch.stack(sorted(outs_over_t[i], key=lambda x:x.max(), reverse=False))
    outs_over_t = outs_over_t.transpose(1,0)
    outs_over_t_list = []
    for i in range(outs_over_t.shape[0]):
        outs_over_t_list.append(outs_over_t[i])
    return outs_over_t_list

def accuracy_over_t(outputs, target, topk=(1,), wordnet_class_threshold = 0.5, sort_by_confidence=True):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        correct_t = []
        preds_over_t = []
        if(sort_by_confidence):
            #print("sorting")
            for output in outputs:
                output = torch.nn.Softmax(dim=-1)(output)
            outputs = get_sorted_max_over_t(outputs) #Sort to get last time step being equivalent to most confident predictions
        #list_of_i_early_correct = []
        for output in outputs[::-1]:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            preds_over_t.append(pred)
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_t.append(correct)
        
        correct_max = torch.max(torch.stack(correct_t), dim=0)#[0]
        correct = correct_max[0]
        correct_ids = correct_max[1]
        
        #print(correct_ids.shape)
        cases_where_earlier_time_steps = list(torch.where(correct_ids[0]>0)) #the 2nd 0 indicates the last time step pred
        
        threshold_earlier_ts = []
        for i_early_t in cases_where_earlier_time_steps[0]:
            correct_t_step = correct_ids[0][i_early_t]
            pred_last = preds_over_t[0][0][i_early_t]
            pred_at_t = preds_over_t[correct_t_step][0][i_early_t]
            if(imnet_sim[pred_last][pred_at_t]>=wordnet_class_threshold): #this means that it might be a potential overlap in class (same object diff. prediction), hence we disregard if a lesser confident output is correct here
                correct[0][i_early_t] = False
                correct[1][i_early_t] = True
                #print('hi')
            else:
                threshold_earlier_ts.append(i_early_t)
        
        res = []
        
        for k in range(maxk): #Fix top5 such that if True in earlier k then false in other ks
            for i in range(k):
                correct[k][correct[i]] = False
            
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        
        return res, cases_where_earlier_time_steps, threshold_earlier_ts