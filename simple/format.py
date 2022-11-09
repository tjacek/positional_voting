import protocols,utils

class OutputDict(dict):
    def __init__(self, arg=[]):
        super(OutputDict, self).__init__(arg)
    
    def show(self):
        for name_i,out_i in self.keys():
            print(f"{name_i},{str(out_i.diff())}")

def format_output(in_path):
    all_outputs={}	
    for path_i in utils.get_paths(in_path):	
        out_i=protocols.read_output(path_i)
        all_outputs[path_i.split('/')[-1]]=out_i
    return OutputDict(all_outputs)

#def line(name_i):

d=format_output("cmc")
d.show()