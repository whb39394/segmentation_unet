import argparse
#import random
import tensorflow as tf

from util import loader as ld
from util import model
from util import repoter as rp
from set_parm import DIR_READ,DIR_ORI,DIR_SEG

def load_dataset(train_rate):
    loader = ld.Loader(dir_original=DIR_ORI,
                       dir_segmented=DIR_SEG)
    return loader.load_train_test(train_rate=train_rate, shuffle=True)

def train(parser):
    # 結果保存用のインスタンスを作成します
    # Create Reporter Object
    reporter = rp.Reporter(parser=parser)
    accuracy_fig = reporter.create_figure("Accuracy", ("epoch", "accuracy"), ["train", "test"])
    loss_fig = reporter.create_figure("Loss", ("epoch", "loss"), ["train", "test"])

    # GPUを使用するか
    # Whether or not using a GPU
    gpu = parser.gpu

    # モデルの生成
    # Create a model
    model_unet = model.UNet(l2_reg=parser.l2reg).model

    # 誤差関数とオプティマイザの設定をします
    # Set a loss function and an optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,
                                                                           logits=model_unet.outputs))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 精度の算出をします
    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # セッションの初期化をします
    # Initialize session
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), device_count={'GPU': 1},
                                log_device_placement=False, allow_soft_placement=True)
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()

    # 不读取时初始化
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    # 恢复数据
    if DIR_READ != 0:
        ckpt = tf.train.get_checkpoint_state(DIR_READ)
        print("********read last model from:********")
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("********create new model********")


    # モデルの訓練
    # Train the model
    epochs = parser.epoch
    batch_size = parser.batchsize
    is_augment = parser.augmentation

    for epoch in range(epochs):
        if epoch % 5 == 0:
            # 是否每次都随机更换一下选取的test图片编号？
            # Load train and test datas
            train, test = load_dataset(train_rate=parser.trainrate)
            valid = train.perm(0, 30)
            test = test.perm(0, 150)
            train_dict = {model_unet.inputs: valid.images_original, model_unet.teacher: valid.images_segmented,
                          model_unet.is_training: False}
            test_dict = {model_unet.inputs: test.images_original, model_unet.teacher: test.images_segmented,
                         model_unet.is_training: False}

        for batch in train(batch_size=batch_size, augment=is_augment):
            # バッチデータの展開
            inputs = batch.images_original
            teacher = batch.images_segmented
            # Training
            sess.run(train_step, feed_dict={model_unet.inputs: inputs, model_unet.teacher: teacher,
                                            model_unet.is_training: True})
        # 評価
        # Evaluation
        if epoch % 1 == 0:
            loss_train = sess.run(cross_entropy, feed_dict=train_dict)
            loss_test = sess.run(cross_entropy, feed_dict=test_dict)
            accuracy_train = sess.run(accuracy, feed_dict=train_dict)
            accuracy_test = sess.run(accuracy, feed_dict=test_dict)
            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train)
            print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)
            accuracy_fig.add([accuracy_train, accuracy_test], is_update=True)
            loss_fig.add([loss_train, loss_test], is_update=True)
        if epoch % 5 == 0:
                #随机展示结果=>因为数据会重新打乱顺序所以固定值即随机
                # idx_train = random.randrange(10)
                # idx_test = random.randrange(3)
                idx_train = 1
                idx_test = 1
                outputs_train = sess.run(model_unet.outputs,
                                         feed_dict={model_unet.inputs: [train.images_original[idx_train]],
                                                    model_unet.is_training: False})
                outputs_test = sess.run(model_unet.outputs,
                                        feed_dict={model_unet.inputs: [test.images_original[idx_test]],
                                                   model_unet.is_training: False})
                train_set = [train.images_original[idx_train], outputs_train[0], train.images_segmented[idx_train]]
                test_set = [test.images_original[idx_test], outputs_test[0], test.images_segmented[idx_test]]
                reporter.save_image_from_ndarray(train_set, test_set, train.palette, epoch,
                                                 index_void=255)
                                                #index_void = len(ld.DataSet.CATEGORY) - 1)
        if epoch % 5 == 0:
            reporter.save_model(saver,sess,epoch)

    # 訓練済みモデルの評価
    # Test the trained model
    loss_test = sess.run(cross_entropy, feed_dict=test_dict)
    accuracy_test = sess.run(accuracy, feed_dict=test_dict)
    print("Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)

    #保存模型+输出所有最终结果
    reporter.save_model(saver,sess,epochs)


    train_image_number = 16
    test_image_number = 4
    for index_n in range(0, train_image_number):
        final_train = sess.run(model_unet.outputs,
                                feed_dict={model_unet.inputs: [train.images_original[index_n]],
                                           model_unet.is_training: False})
        final_train_set = [train.images_original[index_n], final_train[0], train.images_segmented[index_n]]
        reporter.save_all_image_final_train(final_train_set, train.palette, index_n, index_void=255)
    for index_n in range(0, test_image_number):
        final_test = sess.run(model_unet.outputs,
                                feed_dict={model_unet.inputs: [test.images_original[index_n]],
                                           model_unet.is_training: False})
        final_test_set = [test.images_original[index_n], final_test[0], test.images_segmented[index_n]]
        reporter.save_all_image_final_test(final_test_set, test.palette, index_n, index_void=255)

    sess.close()



def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python main.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPUs')
    parser.add_argument('-e', '--epoch', type=int, default=300, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=8, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float, default=0.8, help='Training rate')#train 中 拿出来训练的百分比
    parser.add_argument('-a', '--augmentation',default=False , action='store_true', help='Number of epochs')
    parser.add_argument('-r', '--l2reg', type=float, default=0.001, help='L2 regularization')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
