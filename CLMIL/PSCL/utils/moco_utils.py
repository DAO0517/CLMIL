import os
import torch
import math
import matplotlib.pyplot as plt

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():  # 上下文管理器，禁用梯度计算，以便在计算准确率时不会影响模型的梯度
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  # 前k个结果
        pred = pred.t()
        # 将目标标签target视图扩展为与pred相同的形状，使用eq()函数进行元素级别的比较，生成一个布尔值张量correct，表示预测是否正确
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)  # view（-1）展平为一维张量
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.num_epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.   # 如果没到里程碑节点，那么lr将*1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def visualization(num_epochs, value_list, title, model_saved_path):
    plt.plot(range(num_epochs), value_list[0], 'b-', label=f'Training_{title}')
    title_name = 'Training ' + title 
    
    if len(value_list) == 2:
        plt.plot(range(num_epochs), value_list[1], 'g-', label=f'validation{title}')
        title_name = 'Training & Validation ' + title 
        
    plt.title(title_name)
    plt.xlabel('Number of epochs')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(os.path.join(model_saved_path, f"{title}.jpg"))
    plt.close()
    