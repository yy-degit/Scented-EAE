import logging
from copy import deepcopy


logger = logging.getLogger("__main__")


class Metric:
    def __init__(self):
        self.tp = 0
        self.gold_num = 0
        self.pred_num = 0
        self.p = 0.0
        self.r = 0.0
        self.f1 = 0.0


    def compute(self, targets, outputs):
        self.gold_num = sum([len(one) for one in targets])
        self.pred_num = sum([len(one) for one in outputs])
        for target, output in zip(targets, outputs):
            dup_gold_list = deepcopy(target)
            for pred in output:
                if pred in dup_gold_list:
                    self.tp += 1
                    dup_gold_list.remove(pred)
        self.p = self.safe_div(self.tp, self.pred_num) 
        self.r = self.safe_div(self.tp, self.gold_num) 
        self.f1 = self.safe_div(2 * self.p * self.r, self.p + self.r)
    

    @staticmethod
    def safe_div(a, b):
        if b == 0.0:
            return 0.0
        else:
            return a / b
        
    
    def to_dict(self, prefix):
        return {
            prefix + "-gold_num": self.gold_num,
            prefix + "-pred_num": self.pred_num,
            prefix + "-tp": self.tp,
            prefix + "-p": round(self.p, 4),
            prefix + "-r": round(self.r, 4),
            prefix + "-f1": round(self.f1, 4) 
        }