import protocols,utils,clfs,cv,opv

def cv_exp(in_path="cleveland",out_path='balanced',pairs=None,
	    clf_alg=None,n_iters=10):
    if(pairs is None):
        pairs=[(0,2),(0,3),(0,4)]
    if(clf_alg is None):
        clf_alg=clfs.rf_clf()
    utils.make_dir(out_path)
    for pair_i in pairs:
        selector=cv.BalancedSelector(*pair_i)
        opv_exp=protocols.OPVExp(limit=12,selector=selector)
        output=protocols.multi_exp(in_path,clf_alg,metric=None,
            n_iters=n_iters,opv_exp=opv_exp)
        output.save(f'{out_path}/{pair_i[1]}')

def inspect(in_path):
    for path_i in utils.get_paths(in_path):
        out_i=protocols.read_output(path_i)
        print(f"{path_i}mean:{out_i.mean()}diff{out_i.diff(True)}")

def make_protocol():
    selector=cv.BalancedSelector(0,4)
    opv_exp=protocols.OPVExp(limit=12,selector=selector)
    clf_algs=[clfs.rf_clf(),clfs.bag_clf()]
    metrics=[opv.auc_metric,opv.acc_metric,opv.f1_metric]
    return protocols.Protocol(clf_algs,metrics,opv_exp=opv_exp)

protocol= make_protocol()
protocol("cleveland",'full',n_iters=10)
#inspect("balanced")