import files

def find_result(in_path,cond):
    def helper(root):
        return (root.split('/')[-1]==cond)
    return files.find_paths(in_path,helper)

paths=find_result("B",'raw')
print(paths)