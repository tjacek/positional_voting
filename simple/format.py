import protocols,utils

class OutputDict(dict):
    def __init__(self, arg=[]):
        super(OutputDict, self).__init__(arg)
    
    def show(self,fun=None):
        if(fun is None):
            fun=lambda out_i:out_i.diff(True)
        for name_i,out_i in self.items():
            print(f"{name_i},{fun(out_i):2.4f}")

def format_output(in_path):
    all_outputs={}	
    for path_i in utils.get_paths(in_path):	
        out_i=protocols.read_output(path_i)
        all_outputs[path_i.split('/')[-1]]=out_i
    return OutputDict(all_outputs)

#def line(name_i):

d=format_output("cmc")
d.show()