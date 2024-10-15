import numpy as np
from statistics import mean 


class MRR: 
    
    def __init__(self, length=20):
        self.length = length
        self.MRR_score = []
    def add(self, recommendation_list, next_item):
        
        res = recommendation_list[:self.length]
        if next_item in res.index:
            rank = res.index.get_loc( next_item ) + 1
            self.MRR_score.append(1.0/rank)    
        else:
            self.MRR_score.append(0)
            
    def score(self):
        return mean(self.MRR_score)

class Precision: 
    
    def __init__(self, length=20):
        self.length = length
        self.precision_score = []
        self.totat_sessionsIn_data = 0
        
    def add(self, recommendation_list, next_items):
        next_items = [next_items]
        if len(next_items) > 1:
            pass
        else:
            res = recommendation_list[:self.length]
            TP  = set(next_items) & set(res.index)
            
    
            if len(TP) > 0:
                hit = float( len(TP)    / self.length )
                self.precision_score.append(hit) 
            else:
                hit = 0.0 / self.length
                self.precision_score.append(hit)   
    def score(self):
        return mean(self.precision_score)  
    
class HR: 
    
    def __init__(self, length=20):
        self.length = length
        self.HR_score = []
        self.totat_sessionsIn_data = 0
        
    def add(self, recommendation_list, next_item):
        res = recommendation_list[:self.length]
        if next_item in res.index:
            self.HR_score.append(1.0)
        else:
            self.HR_score.append(0)
        
    def score(self):
        return mean(self.HR_score) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    