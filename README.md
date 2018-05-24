中文讀者請參考我的文章 [Caffe 遷移學習範例： 分辨猿類相片](https://newtoypia.blogspot.tw/2017/09/caffe-transfer-learning.html)。

This is a collection of notes and codes about my successful attempt at transfer learning.

1. Start a [floydhub deep learning docker](https://github.com/floydhub/dl-docker)
2. ```pip install opencv-python lmdb```
3. Create lmdb's from pics: ```./pic2lmdb.py wnid-apes.txt /path/to/ape/pics```
4. Compute mean: ```$CAFFE_ROOT/build/tools/compute_image_mean -backend=lmdb /root/shared/imnet/training/ mean.binaryproto```
5. Train: ```$CAFFE_ROOT/build/tools/caffe train --solver=/root/shared/imnet/ape/solver.prototxt --weights $CAFFE_ROOT/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel ; date) 2>&1 | tee train.log```
6. Find the mean pixel: ```./bpt2npy.py mean.binaryproto mean.npy```
   This will print a 3-element list that looks like '[103.939, 116.779, 123.68]'
   Replace the ```--mean '...'``` values in the following command with these values.
   Why? Visit [VGG's CNN page](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
   and find "mean pixel" within the two "information page" links --
   we don't really need the mean files at classification time.
   Simply having the mean pixel values is good enough.
7. Test on unknown images: ```./cnclassify.py -f csv -c /root/shared/imnet/ape/
   --labels wnid-apes.txt --mean '[103.939, 116.779, 123.68]' --model deploy.prototxt
   --weights _iter_4000.caffemodel```
   Note: By putting all the relevant files in /root/shared/imnet/ape/
   you can leave out the paths and just specify the file names
   for --labels, --model, and --weights .

You may need to change file and directory paths on the
command line as well as in the *.prototxt files.
Try ```./pic2lmdb.py -h``` and ```./cnclassify.py -h```
to see more options.

Code is forked from Adil Moujahid's [deeplearning-cats-dogs-tutorial](https://github.com/adilmoujahid/deeplearning-cats-dogs-tutorial)
and extensively modified to make it more generic.
Also see his wonderful blog post:
[A Practical Introduction to Deep Learning
with Caffe and Python](http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/) explaining the code.

Use vimdiff or similar editors to study the difference
between my *.prototxt files and the corresponding
files from caffe or from Adil Moujahid.
Also see these two tips for parameter setting in *.prototxt:
[Running Over Whole Sets/Computing Epochs Instead of Iterations](https://github.com/BVLC/caffe/issues/1094),
[Choosing batch sizes and tuning sgd](https://github.com/BVLC/caffe/issues/218)

[2018/5/24] See [This job at floydhub](https://www.floydhub.com/ckhung/projects/transfer/70)
for another demo: transfer learning for classifying a few kinds of fruits.
All the code, config, and data are available for you to
reproduce my experiment. Then, suppose you have downloaded
fruit_iter_60.caffemodel from my output and have
shared all the required files (via -v ...:/SH)
into /SH of your local docker, you can create the
validation statistics:
```time python /SH/code/cnclassify.py -f csv
--model /SH/code/deploy.prototxt
--weights /SH/output/fruit_iter_2000.caffemodel
--labels /SH/code/fruit-wnid.txt
$(perl -pe 's#/fruit/#/SH/fruit/#'
/SH/fruit-lmdb/validation/index.txt)
> /SH/code/validation.csv```
Finally generate a table of correct/incorrect
counts by labels: ```grep , validation.csv | cut -d , -f 3- |
sed 's# ##g; s#/SH/fruit/##; s#-.*##' |
python wnidsubst.py -w fruit-wnid.txt |
python tabcc.py```
Note that wnidsubst.py and tabcc.py are newly written
scripts only present in this repo and not found in the floydhub job.

I hope the following illustration is helpful for other
deep learning newbies behind me.
![files needed in the process of caffe transfer learning](tlprocess.svg)

