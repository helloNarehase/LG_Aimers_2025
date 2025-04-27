class label_encode_with_onehot:
    def __init__(self):
        self.label_id = {}
        self.inital_format = {"id" : 0, "count" : 1}
        self.cur_id = 0
    
    def encode(self, datas:list):
        return_labels = []
        for data in datas:
            ch = self.label_id.get(data)
            if ch is not None:
                self.label_id[data]["count"] += 1
                return_labels.append(self.label_id[data]["id"])
            else:
                self.label_id[data] = self.inital_format.copy()
                self.label_id[data]["id"] = self.cur_id
                return_labels.append(self.label_id[data]["id"])
                self.cur_id += 1

        return return_labels
    
    def with_onehot(self, data:list):
        maxi = max(data)+1
        def one_hot(maxi, id):
            oh_format = [0] * maxi
            oh_format[id] = 1

            return oh_format
        return list(map(lambda id: one_hot(maxi, id), data))
        
        
    def __call__(self, datas):
        return self.encode(datas)