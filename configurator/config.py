class Config(object):
    env = 'default'
    backbone = 'resnet50'
    classify = 'softmax'
    num_classes = 51416
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'cross_entropy'

    display = True
    finetune = True

    train_root = '..'
    train_list = '../data/bbs2.txt'
    #val_list = '/data/Datasets/webface/val_data_13938.txt'

    #test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    #test_list = 'test.txt'

    lfw_root = '../lfw_funneled/'
    lfw_test_list = './lfw_test_pair.txt'

    checkpoints_path = 'ckpt'
    load_model_path = 'models/resnet18.pth'
    test_model_path = 'checkpoints/resnet18_110.pth'
    save_interval = 2000

    train_batch_size = 256  # batch size
    test_batch_size = 256

    input_shape = (128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0'
    num_workers = 32  # how many workers for loading data
    print_freq = 10  # print info every N batch
    lfw_test_interval = 500

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 6020
    lr = [(0, 0.02, 0), (500, 0.02, 0), (501, 0.001, 0.8), (24000, 5, 0.8),
            (100000, 5, 0.8), (150000, 0.4, 0.8)]
    momentum = 0.8
    lr_step = 10
    lr_decay = 0.2  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-6
