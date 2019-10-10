"""
Various important parameters of our model and training procedure.
"""


def get_general_params():
    param = {}
    dn = 1
    param['IMG_HEIGHT'] = int(256/dn)
    param['IMG_WIDTH'] = int(256/dn)
    param['obj_scale_factor'] = 1.14/dn
    param['scale_max'] = 1.05  # Augmentation scaling
    param['scale_min'] = 0.90
    param['max_rotate_degree'] = 5
    param['max_sat_factor'] = 0.05
    param['max_px_shift'] = 10
    param['posemap_downsample'] = 2
    param['sigma_joint'] = 7/4.0
    param['n_joints'] = 14
    param['n_limbs'] = 10

    # Using MPII-style joints: head (0), neck (1), r-shoulder (2), r-elbow (3), r-wrist (4), l-shoulder (5),
    # l-elbow (6), l-wrist (7), r-hip (8), r-knee (9), r-ankle (10), l-hip (11), l-knee (12), l-ankle (13)
    param['limbs'] = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [2, 5, 8, 11]]

    param['n_training_iter'] = 200000
    param['test_interval'] = 500
    param['model_save_interval'] = 50000
    param['project_dir'] = '/home/jl5/metapix/posewarp' # replace with yours
    param['model_save_dir'] = '/data/jl5/data-posewarp/models' # " "
    param['data_dir'] = '/home/jl5/data/data-meta/dance_vids/' # " "
    param['batch_size'] = 4
    param['batch_size_test'] = 1
    param['save_img_dir'] = '/home/jl5/data/data-posewarp/saved_imgs/' # " "
    param['epsilon'] = 0.1
    param['meta_iterations'] = 1000
    param['metamodel_save_interval'] = 500
    param['img_path'] = "/home/jl5/data/data-meta/dance_vids/"
    param['keypoints_path_train'] = "/home/jl5/data/dancevidkeypoints_train/"
    param['keypoints_path_test'] = "/home/jl5/data/dancevidkeypoints/"

    return param
