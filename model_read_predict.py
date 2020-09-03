import argparse
import random
import tensorflow as tf

from util import loader as ld
from util import model
from util import repoter as rp


LOAD_MODEL='../result_save/20200826_2058/model'

# DIR_READ="data_set/AR_marker_origin_id/train"
# DIR_SAVE="data_set/AR_marker_origin_id/results/train"
# DIR_SAMPLE="data_set/AR_marker_origin_id/label"
# image_number=20

DIR_READ="data_set/AR_marker_air/test"
DIR_SAVE="data_set/AR_marker_air/results/test"
DIR_SAMPLE="data_set/AR_marker_air/fake_test"
image_number=47

# DIR_READ="data_set/AR_marker_laser-move/test"
# DIR_SAVE="data_set/AR_marker_laser-move/results"
# DIR_SAMPLE="data_set/AR_marker_laser-move/fake_test" #需要同名P模式图像
# image_number=10



def load_predict():
    loader = ld.Loader(dir_original=DIR_READ,
                       dir_segmented=DIR_SAMPLE)
    #loader = ld.LoaderPredict(dir_original=DIR_READ, paths_segmented=DIR_SAMPLE)
    return loader.load_predict_image()

def predict(parser):
    # モデルの生成
    model_unet = model.UNet(l2_reg=parser.l2reg).model
    sess = tf.InteractiveSession()

    # 不读取时初始化
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    # 恢复数据
    ckpt = tf.train.get_checkpoint_state(LOAD_MODEL)
    print(ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    reporter = rp.ReporterPre()
    predict_set = load_predict()
    for index_n in range(0, image_number):
        final_predict = sess.run(model_unet.outputs,
                                feed_dict={model_unet.inputs: [predict_set.images_original[index_n]],
                                           model_unet.is_training: False})
        final_predict_set = [predict_set.images_original[index_n], final_predict[0]]
        reporter.save_predict_image(final_predict_set, predict_set.palette, index_n, DIR_SAVE, index_void=255)

    sess.close()


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python main.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPUs')
    parser.add_argument('-e', '--epoch', type=int, default=2, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=1, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float, default=0.8, help='Training rate')#train 中 拿出来训练的百分比
    parser.add_argument('-a', '--augmentation',default=False , action='store_true', help='Number of epochs')
    parser.add_argument('-r', '--l2reg', type=float, default=0.001, help='L2 regularization')
    return parser

if __name__ == '__main__':
    parser = get_parser().parse_args()
    predict(parser)