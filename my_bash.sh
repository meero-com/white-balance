INPATH='/Users/leo/Downloads/CNP2_wavelet_56_12_epochs8_peacock2enhanced/*.jpg'
OUTPATH='/Users/leo/Downloads/CNP_Juliette_output/'

for filename in $INPATH; do
  NAMENODIR=${filename##*/} # filename without directory
  NAMENOEXT="${NAMENODIR%.*}" # filename without extension
  # SUBNAME="$OUTPATH"sub/"$NAMENOEXT.ass" # subtitle filename
  # echo $SUBNAME

  echo $filename
  # echo $NAMENODIR
  # echo $NAMENOEXT

  python3 custom_run_py3.py $filename $OUTPATH$NAMENODIR models/colorchecker_fold1and2.ckpt models/model_p3.pkl

  echo "\n"


done






# python fc4.py test models/pretrained/colorchecker_fold1and2.ckpt -1 ~/Downloads/ForLeo/img_03.jpg
# python fc4.py test models/pretrained/colorchecker_fold1and2.ckpt -1 ~/Downloads/ForLeo/img_04.jpg
# python fc4.py test models/pretrained/colorchecker_fold1and2.ckpt -1 ~/Downloads/ForLeo/img_05.jpg
# python fc4.py test models/pretrained/colorchecker_fold1and2.ckpt -1 ~/Downloads/ForLeo/img_06.jpg