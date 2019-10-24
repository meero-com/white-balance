import argparse
import cv2
import numpy as np
import tensorflow as tf

from my_fcn import FCN


########################################################################


# IMG_IN = "/Users/leo/Downloads/ingedata.jpg"
# IMG_OUT = "/Users/leo/Downloads/ingedata_out_fc4.jpg"
# MODEL = "/Users/leo/code/fc4/models/pretrained/colorchecker_fold1and2.ckpt"
# SQUEEZE = "/Users/leo/code/fc4/data/squeeze_net/model_p3.pkl"

def get_session():
  import tensorflow as tf
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  return tf.compat.v1.Session(config=config)


########################################################################


def test(model, image_pack_name, output_filename, sq_path):
	with get_session() as sess:
		img = cv2.imread(image_pack_name)
		img = (img / 255.0) ** 2.2 * 65536
		images = [img]

		fcn = FCN(sess=sess, name=model, sq_path=sq_path)
		fcn.load_absolute(model)
		illums, confidence_maps, est = fcn.test_external(images=images, fns=[image_pack_name], sq_path=sq_path)
		
		corrected = np.power(img[:,:,::-1] / 65535 / est[None, None, :] * np.mean(est), 1/2.2)[:,:,::-1]
		cv2.imwrite(IMG_OUT, corrected * 255.0)


########################################################################


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Performs the white balance correction on an image')
	parser.add_argument('input', type=str, help='input image to process')
	parser.add_argument('output', type=str, help='output image')
	parser.add_argument('ckpt', type=str, help='ckpt model path')
	parser.add_argument('squeezenet', type=str, help='squeezenet model path')
	args = parser.parse_args()

	IMG_IN = args.input
	IMG_OUT = args.output
	MODEL = args.ckpt
	SQUEEZE = args.squeezenet

	test(model=MODEL, image_pack_name=IMG_IN, output_filename=IMG_OUT, sq_path=SQUEEZE)
