import pandas as pd
import protocols,utils

class OutputDict(dict):
    def __init__(self, arg=[]):
        super(OutputDict, self).__init__(arg)

    def as_lines(self,show=True,fun=None):
        if(fun is None):
            fun=lambda out_i:f"{out_i.diff(True):2.4f}"
        for name_i,out_i in self.items():
            desc_i=",".join(name_i.split("_"))
            line=f"{desc_i},{fun(out_i)}"
            if(show):
                print(line)
            else:
                yield line

    def as_df(self,fun=None):
        lines=list(self.as_lines(show=False,fun=fun))
        return pd.DataFrame(lines,columns=None)

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
    return output.as_lines(helper)	 

d=format_output("cmc")
print(d.as_df())
#all_diff(d)