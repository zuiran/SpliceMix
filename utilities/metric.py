import math
import torch

class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=True):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.filenames = []

    def add(self, output, target, filename):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        output = check_tensor(output)
        target = check_tensor(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

        self.filenames += filename

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap*100

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count

        precision_at_i /= pos_count if pos_count != 0 else -1
        return precision_at_i

    def value2(self, difficult_examples=True):  # an efficient version of calculating mAP
        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        # if hasattr(torch, "arange"):
        #     rg = torch.arange(1, self.scores.size(0) + 1).float()
        # else:
        #     rg = torch.range(1, self.scores.size(0)).float()

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]

            if difficult_examples:
                diffcicult_index = targets != 0
                scores = scores[diffcicult_index].sigmoid()
                targets = targets[diffcicult_index]
                targets[targets==-1] = 0
                rg = torch.arange(1, scores.size(0)+1).float()

            _, sortind = torch.sort(scores, 0, True)
            truth = targets[sortind]

            # compute true positive sums

            tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)

            # compute average precision
            ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
        return ap

    def mAP(self):
        # AP = self.value()
        AP = self.value2() * 100
        mAP = AP.mean()
        return mAP, AP

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores
        targets = self.targets
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = torch.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1]
        tmp = self.scores
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = torch.zeros(n_class), torch.zeros(n_class), torch.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = torch.sum(targets == 1)
            Np[k] = torch.sum(scores >= 0)
            Nc[k] = torch.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = torch.sum(Nc) / torch.sum(Np)
        OR = torch.sum(Nc) / torch.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = torch.sum(Nc / Np) / n_class
        CR = torch.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP * 100, OR * 100, OF1 * 100, CP * 100, CR * 100, CF1 * 100

def check_tensor(tensor):
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    return tensor

class AverageMeter(object):
    """Computes average value"""

    def __init__(self, name, fmt=':f'):
        super(AverageMeter, self).__init__()
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def add(self, val):
        # val's shape must be []

        self.val = val
        self.sum += val
        self.count += 1

    def average(self):
        self.avg = self.sum / self.count
        return self.avg

    def value(self):
        return self.val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    import time

    ap = AveragePrecisionMeter()
    output = torch.randn((40132, 80))
    time1 = time.time()
    output[output > 0] = 1
    output[output <= 0] = 0
    target = output
    ap.reset()
    ap.add(output[:,:-1], target[:,:-1], '1')
    print(ap.mAP()[0])
    time2 = time.time()
    print(ap.value2().mean())
    time3 = time.time()
    t21 = (time2 - time1) / 60
    t32 = (time3 - time2) / 60
    print(f't21: {t21}m, t32: {t32}m')
    ap.overall_topk(3)


    a = 'pause' # for debug