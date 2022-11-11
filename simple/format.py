import protocols,utils

class OutputDict(dict):
    def __init__(self, arg=[]):
        super(OutputDict, self).__init__(arg)
    
    def show(self,fun=None):
        if(fun is None):
            fun=lambda out_i:f"{out_i.diff(True):2.4f}"
        for name_i,out_i in self.items():
            print(f"{name_i},{fun(out_i)}")

def format_output(in_path):
    all_outputs={}	
    for path_i in utils.get_paths(in_path):	
        out_i=protocols.read_output(path_i)
        all_outputs[path_i.split('/')[-1]]=out_i
    return OutputDict(all_outputs)

def all_diff(output):
    def helper(out_i):
        lines=[ "%.4f-%.4f" % (base_i,opv_i)
            for base_i,opv_i in zip(*out_i.get_acc())]
        return ",".join(lines)
    return output.show(helper)	 
#def line(name_i):

d=format_output("cmc")
all_diff(d)