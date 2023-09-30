import single_layer as sl
import common as c

if __name__ == '__main__':
    import torchvision
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('-r', '--rows', default=5, type=int, help='Rows')
    args.add_argument('-c', '--cols', default=4, type=int, help='Columns (m=rows*columns)')
    args.add_argument('-ph', '--patch-height', default=5, type=int, help='Patch height')
    args.add_argument('-pw', '--patch-width', default=5, type=int, help='Patch width (n=width*height)')
    args.add_argument('-d', '--data-dir', help='Path to dataset')
    args.add_argument('-dt', '--dataset', default="mnist", help="Type of dataset to use")
    args.add_argument('-p', '--preprocess', help="How to convert rgb/greyscale images into sparse binary vectors")
    args.add_argument('-mth', '--method', default='HardWtaL1')

    args = args.parse_args()
    plotter = c.Plot(rows=args.rows, cols=args.cols)
    rand_patch = c.RandPatch(height=args.patch_height, width=args.patch_width, channels=1)
    method_class = {
        'HardWtaL2': sl.HardWtaL2,
        'HardWtaL1': sl.HardWtaL1,
        'HardWtaZeroOrder': sl.HardWtaZeroOrder,
    }
    method_class = method_class[args.method]
    method = method_class(m=plotter.m, n=rand_patch.n)

    if args.dataset == "custom":
        data = c.DatasetDir("../data/imgs" if args.data_dir is None else args.data_dir)
        preprocess = "top=" + str(rand_patch.n // 5) if args.preprocess is None else args.preprocess
    elif args.dataset == "mnist":
        data = torchvision.datasets.MNIST('../../data' if args.data_dir is None else args.data_dir, train=False, download=True)
        data = data.data.numpy()
        preprocess = "thresh=0.8" if args.preprocess is None else args.preprocess
    else:
        raise Exception("Unknown dataset type " + args.dataset)

    preprocess_function, preprocess_param = preprocess.split("=")
    if preprocess_function == "top":
        preprocess = c.SampleOfCardinality(int(preprocess_param))
    elif preprocess_function == "thresh":
        preprocess = c.SampleByThreshold(float(preprocess_param))
    else:
        raise Exception("Unknown preprocessing function "+preprocess_function)
    trainer = c.Trainer(method, data, rand_patch, preprocess=preprocess)

    while True:
        trainer.train(500, 4)
        means, counts = trainer.eval(len(data), max(4, (plotter.m * 100) // len(data)))
        print("probabilities=", counts / counts.sum())
        plotter.plot(means)
